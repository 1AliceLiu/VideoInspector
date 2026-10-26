"""
DeepStream 探针 → 告警状态机桥（配置驱动）

把两阶段检测（person 锚点整帧检测 + detector 整帧 + classifier 二级分类）的 batch
元数据翻译成 server.metadata.ObjectMeta 列表，按 source_id 喂给对应的 AlertManager；
同时决定**预览**的 OSD 渲染内容（nvdsosd 原生渲染）。证据帧是原始帧，违规框由
AlertManager 在编码前自画，与此处 OSD 策略无关（见 pipeline/frame_cache 与 alert/manager）。

渲染策略（OSD policy）:

    person     → 应用控制：违规 = 红框 + 违规标签；合规 / 低置信度 = 隐藏
    非 person  → 一律无条件隐藏（无论其规则本路是否激活）

注意：所有 GIE 对每一路都会推理，`active_rules` 只决定「本路判定/告警哪些规则」，并不
决定「哪些模型不推理」。若按 active_rules 决定隐藏哪些 detector，未激活规则的 detector
对象会保留 nvinfer 默认 OSD（border_width=3 + obj_label），在 nvdsosd 上泄漏出框
（如 smoking-only 路泄漏 helmet 框）。因此「是否参与推理」与「是否允许上屏」必须彻底
解耦：非 person 无条件隐藏。

nvdsosd 默认会把每个对象按默认色画出来，因此**不画 = 显式隐藏**
（rect_params.border_width=0 + 清空 text_params.display_text——nvosd 对未设置 text 的
对象会自动显示 obj_label/分类器标签）。此策略只作用于实时预览。

nvosd 经验值：违规框 border_width 实测 2 不上屏、4 起正常渲染，取 4（1080p 下粗细
与辨识度均衡，GPU 光栅化对小宽度有下限）。

模型拓扑来自 config（model.gies + rules），探针不再硬编码任何 uid/规则名：
  - detector（如 helmet）：整帧检测器，空间关联（检测框中心落在 person 框内且
    label == violation）→ 违规
  - classifier（如 harness/vest）：读 person 对象上的 NvDsClassifierMeta，label ==
    violation → 违规
  - 违规翻译为 AttributeMeta(rule_name)，rule_name 即 alert_type。

空间作用域（zones）与命中语义（trigger）:
  - 规则带 zones 时，**区域门挂在属性上**：violation 型先判违规、再要求人员锚点落在
    区域内，否则丢弃该属性；未带 zones = 全图判定（现有 PPE 算法的默认形态）。
  - trigger=presence（区域入侵）不看任何违规属性，「区域内出现人员」本身即命中，
    置信度取 person 检出分；该型规则必须绑定区域，由 validate_config 在启动期拦截。
  - 区域判定复用 person 检出结果，**不新增任何模型推理**；GPU 开销与全图判定一致，
    变的只是「哪些命中进入告警状态机」。
  - 锚点默认 feet（底边中点）；归一化坐标 → 像素由 mux 尺寸换算（见 alert/regions）。

约束（与旧版一致）:
  - 本探针在 GStreamer 流线程运行，只做轻量决策；JPEG/base64/HTTP 由 AlertManager
    内部 executor + daemon 线程处理。
  - 隐藏/上色必须在同一 pass 内完成（object_items 是一次性迭代器，不能跨 pass 持有包装器）。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from pyservicemaker import BatchMetadataOperator, osd
from loguru import logger

from server.metadata import AttributeMeta, ObjectMeta
from server.model_spec import KIND_CLASSIFIER, KIND_DETECTOR
from server.alert.rules import RuleConfig, TRIGGER_PRESENCE
from server.alert.regions import ZoneMatcher, anchor_point

RED = osd.Color(1.0, 0.0, 0.0, 1.0)    # 违规（唯一上屏色）

# 隐藏对象的两件事：边宽置 0（nvosd 不画框）+ 显式清空 display_text
# （nvosd 对未设置 text 的对象会自动显示 obj_label，如 helmet/person，必须显式压掉）
_HIDE_BOX = 0
_HIDE_TEXT = b""

# 分类器置信度解析失败时的回退值（payload 展示用，判定由 classifier-threshold 负责）
CLASSIFIER_FALLBACK_CONF = 0.5


class SafetyProbe(BatchMetadataOperator):
    """逐帧: 按激活规则计算违规 → ObjectMeta 列表 → 对应摄像头的 AlertManager.handle()。

    Args:
        alert_managers: {source_id(int): AlertManager}，每路摄像头一个状态机。
        executor: ThreadPoolExecutor，透传给 AlertManager 用于异步 webhook。
        frame_cache: 可选 FrameCache，触发告警时取该 source 最新已渲染帧作 snapshot。
        gies: {gie_name: GieSpec}，模型拓扑（来自 config）。
        rules: {rule_name: RuleConfig}，告警规则（gie 引用 gies）。
        person_uid: 锚点检测器（person）的 uid。
        person_conf_threshold: person 置信度门槛（低于视为噪声）。
        active_rules_by_source: {source_id: {rule_name}}，每路摄像头激活的规则。
    """

    def __init__(self, alert_managers: dict,
                 executor: ThreadPoolExecutor | None = None,
                 frame_cache=None,
                 gies: dict | None = None,
                 rules: dict[str, RuleConfig] | None = None,
                 person_uid: int = 1,
                 person_conf_threshold: float = 0.6,
                 active_rules_by_source: dict[int, set[str]] | None = None,
                 health=None,
                 mux_width: int = 1920,
                 mux_height: int = 1080,
                 log_out_of_zone: bool = False):
        super().__init__()
        self._managers = alert_managers
        self._executor = executor
        self._frame_cache = frame_cache
        self._health = health   # SourceHealthMonitor（可选）: 逐路帧计数供看门狗判定
        self._gies = gies or {}
        self._rules = rules or {}
        self._person_uid = person_uid
        self._person_conf_threshold = person_conf_threshold
        self._active_rules_by_source = active_rules_by_source or {}
        self._all_rules = set(self._rules)
        # 区域标定期排障开关：记录「违规成立但人在区域外被淘汰」的命中。
        # 默认关闭（热路径零开销）；配了区域的路上打开，可现场确认
        # 「不报」到底是模型没检出，还是检出后被区域门挡下。
        self._log_out_of_zone = log_out_of_zone

        # 预解析: rule → gie 规格（uid / kind / violation）
        # detector 与 classifier 分桶，热路径免查 kind。
        self._rule_binding: dict[str, tuple] = {}   # rule_name -> (uid, kind, violation)
        self._known_labels: set[str] = set()
        # 区域命中器（rule → ZoneMatcher）与 presence 型规则集合：
        #   - 有 zones 的规则才建 matcher（无 zones = 全图，不进区域判定，零开销）
        #   - presence 型只看「区域内有没有人」，不看任何违规属性
        self._matchers: dict[str, ZoneMatcher] = {}
        self._presence_rules: set[str] = set()
        for rule_name, rule in self._rules.items():
            spec = self._gies[rule.gie]
            self._rule_binding[rule_name] = (spec.uid, spec.kind, spec.violation)
            if spec.violation:
                self._known_labels.add(spec.violation)
            if rule.trigger == TRIGGER_PRESENCE and rule.zones:
                self._presence_rules.add(rule_name)
            if rule.zones:
                self._matchers[rule_name] = ZoneMatcher(rule.zones,
                                                        mux_width, mux_height)
            if rule.trigger == TRIGGER_PRESENCE and not rule.zones:
                # 两种情形要分开说，否则启动日志会误导排障：
                #   ① 模板规则（尚未有摄像头绑定区域）→ 正常，等导入器补上
                #   ② 已被某路 active_rules 引用却没区域 → 配置有问题（validate_config
                #      本应在启动期拦住，走到这里说明有旁路），该规则全程不命中
                referenced = (set().union(*self._active_rules_by_source.values())
                              if self._active_rules_by_source else self._all_rules)
                if rule_name in referenced:
                    logger.warning("规则 {} 是 presence 型但被摄像头引用且未绑定区域，"
                                   "本次启动全程不生效（区域入侵未标定？）", rule_name)
                else:
                    logger.info("规则 {} 是 presence 模板，暂未绑定区域且未被任何路引用"
                                "（标定后即生效）", rule_name)

    def handle_metadata(self, batch_meta):
        health_record = self._health.record if self._health is not None else None
        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            if health_record is not None:
                health_record(source_id)
            active = self._active_rules_by_source.get(source_id, self._all_rules)

            active_detector_uids: set[int] = set()
            for rule_name in active:
                uid, kind, violation = self._rule_binding[rule_name]
                # presence 型规则挂的是锚点(person)本身，不参与「整帧检测器空间关联」
                # 收集；否则 person uid 会进这个集合却被下面第一趟 continue 跳过，
                # 白建一个永远为空的桶。
                if kind == KIND_DETECTOR and violation is not None:
                    active_detector_uids.add(uid)

            # 第一趟: OSD policy —— 非 person 一律无条件隐藏（与 active_rules 解耦）。
            # 同时收集本路激活 detector 的普通数值（label/conf/中心点）供 person 空间
            # 关联；object_items 是一次性迭代器，隐藏与收集必须同一趟完成。
            detector_boxes: dict[int, list] = {uid: [] for uid in active_detector_uids}
            for o in frame_meta.object_items:
                uid = o.unique_component_id
                if uid == self._person_uid:
                    continue
                o.rect_params.border_width = _HIDE_BOX
                o.text_params.display_text = _HIDE_TEXT
                if uid in active_detector_uids:
                    detector_boxes[uid].append((
                        getattr(o, "label", ""),
                        o.confidence,
                        o.rect_params.left + o.rect_params.width / 2,
                        o.rect_params.top + o.rect_params.height / 2,
                    ))

            # 第二趟: 每个 person → ObjectMeta（违规进 attributes）+ 渲染决策
            objects: list[ObjectMeta] = []
            for obj in frame_meta.object_items:
                if obj.unique_component_id != self._person_uid:
                    continue
                if obj.confidence < self._person_conf_threshold:
                    obj.rect_params.border_width = _HIDE_BOX  # 噪声框/字也须显式隐藏
                    obj.text_params.display_text = _HIDE_TEXT
                    continue

                # 代表点只算一次（feet/center 各一个浮点对），供所有区域规则复用
                bbox = (
                    int(obj.rect_params.left),
                    int(obj.rect_params.top),
                    int(obj.rect_params.left + obj.rect_params.width),
                    int(obj.rect_params.top + obj.rect_params.height),
                )
                feet = anchor_point(*bbox, "feet")
                center = anchor_point(*bbox, "center")

                attrs: list[AttributeMeta] = []
                for rule_name in active:
                    rule = self._rules[rule_name]
                    uid, kind, violation = self._rule_binding[rule_name]

                    # ① presence 型（区域入侵）: 区域内出现人员即命中，不看任何违规属性
                    if rule_name in self._presence_rules:
                        zone = self._matchers[rule_name].hit(feet, center)
                        if zone is not None:
                            attrs.append(AttributeMeta(rule_name, obj.confidence,
                                                       zone=zone.id))
                        continue

                    # ② violation 型: 先判违规（现有逻辑），再过区域门
                    if kind == KIND_DETECTOR:
                        conf = self._match_violation(
                            obj, detector_boxes.get(uid, []),
                            violation, rule.attribute_threshold,
                        )
                    else:  # classifier
                        label = self._classifier_label(obj.classifier_items, uid)
                        conf = CLASSIFIER_FALLBACK_CONF if label == violation else None

                    if conf is None:
                        continue

                    matcher = self._matchers.get(rule_name)
                    if matcher is not None:
                        zone = matcher.hit(feet, center)
                        if zone is None:
                            # 违规成立但人在区域外 → 不告警。现场标定期极易被误读成
                            # 「模型没跑」，故留一条可按路开关的 DEBUG（默认关）。
                            if self._log_out_of_zone:
                                logger.debug("out_of_zone camera={} rule={} conf={:.2f}",
                                             self._camera_name(source_id), rule_name, conf)
                            continue
                        attrs.append(AttributeMeta(rule_name, conf, zone=zone.id))
                    else:
                        attrs.append(AttributeMeta(rule_name, conf))

                # 有违规=红框+标签；无违规=隐藏（classifier 合规类不可观测）
                if attrs:
                    obj.rect_params.border_width = 4  # nvosd 怪癖: 2 不上屏，见模块 docstring
                    obj.rect_params.border_color = RED
                    # 对外显示基础算法名，避免 __cam__ 内部规则名漏到预览
                    labels = [self._rules[a.name].alert_type for a in attrs]
                    self._set_osd_label(obj, " ".join(labels))
                else:
                    obj.rect_params.border_width = _HIDE_BOX
                    obj.text_params.display_text = _HIDE_TEXT

                objects.append(ObjectMeta(
                    class_name="person",
                    confidence=obj.confidence,
                    bbox=bbox,
                    attributes=tuple(attrs),
                ))

            manager = self._managers.get(source_id)
            if manager is not None:
                snapshot, frame_info = None, None
                if self._frame_cache is not None:
                    entry = self._frame_cache.latest_with_meta(source_id)
                    if entry is not None:
                        snapshot, snap_fn, snap_ts = entry
                        # decision 侧取本帧元数据；snapshot 侧取缓存帧，供告警时打
                        # evidence 同步日志（delta_fn / delta_ms），用于验证证据帧滞后。
                        frame_info = (
                            getattr(frame_meta, "frame_number", -1),
                            getattr(frame_meta, "buffer_pts", 0),
                            snap_fn,
                            snap_ts,
                        )
                manager.handle(objects, snapshot=snapshot,
                               executor=self._executor, frame_info=frame_info)
            elif objects:
                logger.debug("source_id={} 无对应 AlertManager，跳过告警判定", source_id)

    # ------------------------------------------------------------------
    # 静态工具
    # ------------------------------------------------------------------
    def _camera_name(self, source_id: int) -> str:
        """日志用路名（AlertManager 未注册时回退 source_id，不影响判定）。"""
        manager = self._managers.get(source_id)
        return getattr(manager, "camera_name", None) or f"source{source_id}"

    @staticmethod
    def _match_violation(obj, boxes, violation_label, conf_threshold):
        """中心点落在 person 框内、label==violation、置信度达标的框 → 最高置信度（无则 None）。"""
        px1 = obj.rect_params.left
        py1 = obj.rect_params.top
        px2 = px1 + obj.rect_params.width
        py2 = py1 + obj.rect_params.height
        best = None
        for label, conf, cx, cy in boxes:
            if label != violation_label or conf < conf_threshold:
                continue
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                if best is None or conf > best:
                    best = conf
        return best

    @staticmethod
    def _set_osd_label(o, text: str):
        if not text:
            return
        o.text_params.display_text = text.encode("ascii")
        o.text_params.x_offset = int(o.rect_params.left)
        o.text_params.y_offset = max(0, int(o.rect_params.top) - 6)
        o.text_params.font_params.name = osd.FontFamily.Serif
        o.text_params.font_params.size = 12
        o.text_params.font_params.color = RED

    # 每个 uid 只 DEBUG 一次 get_n_label 原始格式（避免热路径逐人逐帧刷屏）
    _logged_label_formats: set[int] = set()

    @classmethod
    def _classifier_label(cls, classifier_items, uid: int) -> str | None:
        """读 person 对象的二级分类器结果 → 首个挂载的 label 名（class0，即违规类）。"""
        try:
            for clf in classifier_items:
                if getattr(clf, "unique_component_id", None) != uid:
                    continue
                for i in range(getattr(clf, "n_labels", 0)):
                    raw = clf.get_n_label(i)
                    label, _conf = cls._parse_classifier_label(raw)
                    if uid not in cls._logged_label_formats:
                        cls._logged_label_formats.add(uid)
                        logger.debug("classifier uid={} raw={!r} parsed={}", uid, raw, label)
                    return label
        except Exception:
            logger.exception("读取分类器结果异常")
        return None

    @staticmethod
    def _parse_classifier_label(raw: str):
        """防御性解析 get_n_label 字符串 → (label_name | None, confidence | None)。"""
        if not raw:
            return None, None
        tokens = re.split(r"[: \t]+", raw.strip())
        conf = None
        for t in tokens:
            try:
                conf = float(t)
            except ValueError:
                pass
        # label 取第一个非数值 token（get_n_label 只挂 class0，即违规类）
        for t in tokens:
            try:
                float(t)
            except ValueError:
                return t, conf
        return None, conf
