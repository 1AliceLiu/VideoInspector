"""
告警管理器
- 显式状态机按「规则」独立（每摄像头每规则一套），规则之间互不竞争:
    每条规则独立 冷却窗口 / 连续违规帧计数 / 违规置信度门限
- 连续帧确认：连续 N 帧检测到才触发告警（防止单帧误报）
- 告警分发：获取证据帧快照（已由 nvdsosd 原生渲染）→ executor 线程 JPEG 编码 →
           daemon 线程 fire-and-forget 完成 base64/JSON/HTTP 推送
  调用线程仅做决策，不做图片。
"""

import base64
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum

import cv2
import numpy as np
import simplejpeg
from loguru import logger

from server.metadata import ObjectMeta
from server.alert import vlm_review
from server.alert.regions import Zone
from server.alert.rules import RuleConfig, TRIGGER_PRESENCE
from server.alert.webhook import WebhookAlerter


class AlertState(Enum):
    """告警状态机的显式状态。"""
    IDLE = "idle"           # 无近期违规，待触发
    ARMING = "arming"       # 正在累计连续违规帧
    COOLDOWN = "cooldown"   # 告警后冷却窗口，期间不再判定


class _RuleState:
    """单条 violation 型规则的独立状态机（冷却 / 连续帧确认）。

    与摄像头解耦：同摄像头多条规则各自演进，互不竞争。
    """

    def __init__(self, rule: RuleConfig):
        self.rule = rule
        self._state = AlertState.IDLE
        self._consecutive_hits = 0
        self._cooldown_until = 0.0  # monotonic 时间戳：冷却结束时刻

    def handle(self, alert_objects: list[ObjectMeta]) -> list[ObjectMeta] | None:
        """处理一帧的本规则命中结果。

        Args:
            alert_objects: 携带本规则违规属性的 ObjectMeta 列表（空=无违规）。

        Returns:
            达到连续帧阈值时返回触发对象列表并进入冷却，否则 None。
        """
        now = time.monotonic()

        # COOLDOWN：冷却期内不做任何判定/计数，过期后回到 IDLE 重新武装
        if self._state == AlertState.COOLDOWN:
            if now < self._cooldown_until:
                return None
            self._state = AlertState.IDLE
            self._consecutive_hits = 0

        if alert_objects:
            self._state = AlertState.ARMING
            self._consecutive_hits += 1
            if self._consecutive_hits >= self.rule.min_detection_count:
                self._state = AlertState.COOLDOWN
                self._cooldown_until = now + self.rule.cooldown_seconds
                self._consecutive_hits = 0
                return alert_objects
        else:
            # 无违规 → 衰减（每帧 -1，容忍推理偶发丢帧），减到 0 回 IDLE
            if self._state == AlertState.ARMING:
                self._consecutive_hits -= 1
                if self._consecutive_hits <= 0:
                    self._consecutive_hits = 0
                    self._state = AlertState.IDLE
        return None


class _PresenceState:
    """presence 型规则（区域入侵）的占用状态机 —— 报「事件」，不报「电平」。

    只回答一个问题：「区域内现在有没有人？」

        空 ──(连续 min_detection_count 帧有人)──▶ 占用  ← 此刻发一条告警
        占用 ──(连续 vacant_frames 帧无人)────▶ 空      ← 事件结束，复位待下一次进入

    与 _RuleState 的关键差别（这是区域入侵必须换语义的原因）:
      - _RuleState 在冷却期内**完全不看画面**（直接 return None），所以一波入侵的
        安静期内发生的**下一次**入侵会被吞掉（漏报）；而人一直待在场内时又会每个
        安静周期重复报一次（刷屏）。调大调小都只会在两个毛病之间来回。
      - 本状态机每帧都看画面，只是「状态跳变才报」：一次入侵 = 一条告警；
        人在里面待 30 分钟 = 仍然一条；人走光后再进 = 新的一条。
      - 不需要 tracker：判定量是「区域有没有人」这个布尔值，不涉及人员身份。

    cooldown_seconds 在此**退化为「并入窗口」**（presence 规则建议配 0）：
      - 0 = 关闭。两次告警的最小间隔由两个去抖自然提供（进入 min_detection_count
        帧 + 离开 vacant_frames 帧），不会漏掉任何一次独立入侵。
      - >0 = 地板内的新事件**并入**上一条不单独成单：人员持续在场时会在地板解除的
        那一帧补报；若人在地板解除前就离开了，则该次不再成单（视为上一次事件的余波）。
      所以这个值是用来「合并同一波次进出」的，不是限速器；拿它当限速器会吞掉
      独立的第二次入侵（见 tests 场景 B）。
    """

    def __init__(self, rule: RuleConfig):
        self.rule = rule
        self._occupied = False        # 是否已就「本次占用事件」报过警
        self._hits = 0                # 进入去抖计数
        self._vacant = 0              # 离开去抖计数
        self._floor_until = 0.0       # monotonic：防洪地板结束时刻

    def handle(self, alert_objects: list[ObjectMeta]) -> list[ObjectMeta] | None:
        """处理一帧「区域内是否有人」的采样。

        Args:
            alert_objects: 本帧落在本规则区域内的 ObjectMeta 列表（空=区域内无人）。

        Returns:
            仅在「空 → 占用」的上升沿返回触发对象列表，其余一律 None。
        """
        now = time.monotonic()

        if alert_objects:
            self._vacant = 0
            if self._occupied:
                return None                       # 事件进行中：不再重复报
            self._hits += 1
            if self._hits < self.rule.min_detection_count:
                return None                       # 进入去抖未达标
            if now < self._floor_until:
                return None                       # 并入窗口内：本次不单独成单
            self._occupied = True
            self._hits = 0
            self._floor_until = now + self.rule.cooldown_seconds
            return alert_objects

        self._hits = 0
        if self._occupied:
            self._vacant += 1
            if self._vacant >= self.rule.vacant_frames:
                self._occupied = False            # 连续够久无人 → 事件结束
                self._vacant = 0
        return None


class AlertManager:
    """管理单路摄像头的告警状态与分发（内部按规则拆分独立状态机）。

    检测线程仅做轻量决策（按规则过滤 + 状态迁移）。
    JPEG 编码在 executor 线程执行，base64/JSON/HTTP POST
    由独立 daemon 线程 fire-and-forget，不做任何等待。
    """

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        rules: list[RuleConfig],
        webhook: WebhookAlerter | None = None,
        device_code: str | None = None,
        channel_no: str | None = None,
        include_frame: bool = True,
        vlm: dict | None = None,
        frame_cache=None,
        source_id: int | None = None,
    ):
        """
        Args:
            camera_id: 摄像头唯一 ID
            camera_name: 摄像头显示名称
            rules: 本路激活的告警规则（每条独立状态机，alert_type=规则名）
            webhook: Webhook 推送实例，None 则不推送
            device_code: ICC 设备编码（可选，随 webhook 输出）
            channel_no: ICC 通道号（可选，随 webhook 输出）
            vlm: config vlm 节（enabled 时生效，见 alert/vlm_review.py）
            frame_cache: 证据帧缓存（VLM 复核连拍采样用）
            source_id: 本路在管线中的流序号（查 frame_cache 用）
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.device_code = device_code
        self.channel_no = channel_no
        self.include_frame = include_frame
        self.webhook = webhook
        # VLM 复核（可选）：vlm.enabled 且规则在 vlm.rules 内才走复核门；
        # frame_cache/source_id 缺失（如单测外的旧调用方）时复核不可用，自动回退直推。
        self.vlm = vlm if vlm and vlm.get("enabled") else None
        self.frame_cache = frame_cache if self.vlm is not None else None
        self.source_id = source_id
        # 最近一次告警（供控制面状态查询；只在触发时写，热路径无额外开销）
        self.last_alert_at: str | None = None
        self.last_alert_type: str | None = None
        # 本帧检测器上报的全部目标（vlm-audit 巡检比对用；每帧一次赋值，开销可忽略）
        self.last_objects: list[ObjectMeta] = []

        # 每条规则一个独立状态机（rule_name → 状态机实例）：
        #   violation 型（PPE 类）→ _RuleState（连续帧确认 + 冷却）
        #   presence  型（区域入侵）→ _PresenceState（占用上升沿，报事件不报电平）
        self._rule_states: dict[str, _RuleState | _PresenceState] = {
            rule.name: (_PresenceState(rule) if rule.trigger == TRIGGER_PRESENCE
                        else _RuleState(rule))
            for rule in rules
        }

    # ------------------------------------------------------------------
    # 主入口：每帧调用（检测线程，仅做决策，不做图片）
    # ------------------------------------------------------------------
    def handle(self, objects: list[ObjectMeta],
               snapshot=None, executor=None, frame_info=None) -> bool:
        self.last_objects = objects
        """
        处理一帧的检测结果，按规则独立判定是否触发告警。

        本方法在检测线程中运行，仅执行轻量决策逻辑。
        所有渲染（已由 nvdsosd 完成）/编码/网络 I/O 均通过 executor 异步卸载。

        Args:
            objects: 探针翻译层产出的 ObjectMeta 列表
            snapshot: 证据帧（nvdsosd 已渲染的 BGR numpy 数组），None 表示暂无证据帧
            executor: 可选 ThreadPoolExecutor，用于异步执行 webhook 推送
            frame_info: 可选 (decision_fn, decision_ts, snapshot_fn, snapshot_ts)，
                仅用于触发时打 evidence 同步日志（decision 与 snapshot 的帧/PTS 差）

        Returns:
            True 表示本帧至少触发了一条规则
        """
        triggered = False
        for rule_name, state in self._rule_states.items():
            alert_type = state.rule.alert_type or rule_name
            # 只挑携带本规则违规属性的对象
            alert_objects = [
                obj for obj in objects
                if any(attr.name == rule_name for attr in obj.attributes)
            ]
            if alert_objects:
                attr_names = [
                    ", ".join(a.name for a in obj.attributes)
                    for obj in alert_objects
                ]
                logger.debug("detect camera={} rule={} objs={}", self.camera_name,
                             alert_type, attr_names)
            fired = state.handle(alert_objects)
            if fired:
                self._trigger(rule_name, alert_type, fired, snapshot, executor, frame_info)
                triggered = True
        return triggered

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _trigger(self, rule_name: str, alert_type: str, alert_objects: list[ObjectMeta],
                 snapshot, executor=None, frame_info=None):
        """触发告警：轻量调度 → 卸载所有重操作到 executor。

        本方法在检测线程中运行，仅做日志记录和状态更新。
        JPEG 编码 / base64 / JSON / HTTP POST 全部在 _build_and_send 中执行。

        Args:
            rule_name: 内部规则名（可能是按摄像头生成的 __cam__ 规则）
            alert_type: 对外报警类型（webhook alert_type，通常是基础算法名）
            alert_objects: 触发告警的 ObjectMeta 列表
            snapshot: 证据帧（nvdsosd 已渲染的 BGR numpy 数组），None 表示暂无证据帧
            executor: 可选 ThreadPoolExecutor
            frame_info: 可选 (decision_fn, decision_ts, snapshot_fn, snapshot_ts)
        """
        now = datetime.now(timezone.utc)
        iso_timestamp = now.isoformat()
        self.last_alert_at = iso_timestamp
        self.last_alert_type = alert_type

        best_obj = max(alert_objects, key=lambda o: o.confidence)
        logger.info("🚨 alert camera={} rule={} class={} conf={:.2f}",
                    self.camera_name, alert_type, best_obj.class_name,
                    best_obj.confidence)

        # 证据帧同步观测：decision（判定帧）与 snapshot（缓存帧）的帧号/PTS 差，
        # 用于验证「告警已判定但证据帧无框」是否源于帧滞后（Phase 1 只观测不补框）。
        if frame_info is not None:
            self._log_evidence_sync(alert_type, frame_info)

        # JPEG 编码在 executor 线程，base64/JSON/HTTP 由 daemon 线程 fire-and-forget。
        # snapshot 为 None 时仍推送（frame_base64=null），保证无证据帧时不丢告警。
        # 命中 vlm.rules 的规则改走大模型复核线程（内部自取 executor/webhook 推送）。
        if self.webhook and self._vlm_review_enabled(rule_name):
            threading.Thread(
                target=self._vlm_review,
                args=(rule_name, alert_type, alert_objects, snapshot, iso_timestamp),
                daemon=True,
                name=f"vlm-review-{self.camera_id}-{alert_type}",
            ).start()
        elif self.webhook and executor is not None:
            executor.submit(
                self._build_and_send,
                rule_name, alert_type, snapshot, alert_objects, iso_timestamp,
            )

    def _log_evidence_sync(self, alert_type: str, frame_info):
        """打一条 evidence 同步日志（camera_id / rule / 帧号与 PTS 差）。

        仅观测，不改变判定与取证行为；delta 为负或元信息缺失时相应字段为 None。
        """
        decision_fn, decision_ts, snap_fn, snap_ts = frame_info
        delta_fn = (decision_fn - snap_fn) if (decision_fn >= 0 and snap_fn >= 0) else None
        delta_ms = (
            round((decision_ts - snap_ts) / 1e6, 1)
            if decision_ts and snap_ts else None
        )
        logger.info(
            "evidence camera={} rule={} decision_fn={} snapshot_fn={} "
            "delta_fn={} decision_ts={} snapshot_ts={} delta_ms={}",
            self.camera_id, alert_type, decision_fn, snap_fn,
            delta_fn, decision_ts, snap_ts, delta_ms,
        )

    # ------------------------------------------------------------------
    # VLM 复核（告警二次确认：连拍四宫格问大模型 yes/no，见 alert/vlm_review.py）
    # ------------------------------------------------------------------
    def _vlm_review_enabled(self, rule_name: str) -> bool:
        return (self.frame_cache is not None
                and rule_name in (self.vlm.get("rules") or ()))

    def _vlm_review(self, rule_name: str, alert_type: str, alert_objects: list[ObjectMeta],
                    snapshot, iso_timestamp: str):
        """独立 daemon 线程：触发时刻后按规则采连拍人像（吸烟4帧/防护1帧）→ 问大模型 → 按判定推送。

        采样等待（≤ deadline）与 VLM HTTP 全部阻塞在本线程，不占检测/executor 线程。
        每规则的问题/确认方向/帧数见 vlm_review.REVIEWS。两问合一步骤后三条规则极性统一：
        大模型答 yes = 违规坐实；答 no（含"非真实人员"）= 排除。
        shadow=true（灰度观察）：任何判定都照常推送且 payload 不附 vlm 字段，结论只进日志；
        shadow=false：判定与违规方向相反 → 丢弃；坐实违规或（fail_open 时）不可用 →
        走 _build_and_send 正常推送，payload 附 vlm 字段（verdict/model/frames）供下游审计。
        """
        vlm = self.vlm or {}
        shadow = bool(vlm.get("shadow", False))
        mode = vlm_review.RULE_MODE.get(rule_name)
        if mode is None:   # 问不对问题不如不问：未定义复核方式的规则直推
            logger.warning("vlm-review 规则={} 未在 RULE_MODE 定义，直接推送", rule_name)
            self._build_and_send(rule_name, alert_type, snapshot, alert_objects,
                                 iso_timestamp)
            return
        kind, expect, nframes = mode
        url, model = vlm.get("url", ""), vlm.get("model", "")
        timeout = float(vlm.get("timeout", 5))
        persons, crops, verdict = [], [], None
        t0 = t_sample = time.monotonic()
        try:
            if kind == "burst":        # 吸烟：连拍宫格单问（极性天然简单，垃圾框答 no 自排除）
                best = max(alert_objects, key=lambda o: o.confidence)
                crops = vlm_review.collect_frames(
                    self.frame_cache, self.source_id, best.bbox,
                    frames=nframes,
                    sample_s=float(vlm.get("sample_ms", 120)) / 1000.0,
                    deadline_s=float(vlm.get("deadline_ms", 1500)) / 1000.0,
                    pad=float(vlm.get("pad", 0.15)),
                )
                t_sample = time.monotonic()
                if crops:
                    verdict = vlm_review.ask(url, model, vlm_review.build_grid(crops),
                                             vlm_review.PROMPT, timeout)
            else:                      # PPE：逐人复核（一图多人全进名单），全图红框判人+部位特写判穿戴
                frame = snapshot if isinstance(snapshot, np.ndarray) else None
                if frame is None:
                    ef = self.frame_cache.latest(self.source_id)
                    frame = ef.bgr() if ef is not None else None
                if frame is not None:
                    for obj in sorted(alert_objects, key=lambda o: -o.confidence)[:4]:
                        person, wearing, violated, ms = vlm_review.judge_ppe(
                            url, model, frame, obj.bbox, rule_name, timeout)
                        persons.append({"conf": round(obj.confidence, 2),
                                        "person": person, "wearing": wearing,
                                        "violated": violated, "ms": ms})
                t_sample = time.monotonic()
                asked = [p for p in persons
                         if p["person"] is not None and p["wearing"] is not None]
                if asked:
                    verdict = "yes" if any(p["violated"] for p in asked) else "no"
                # asked 为空 = 大模型全程不可用 → verdict 保持 None → 走 fail_open
        except Exception:
            logger.exception("vlm-review 流程异常 camera={}", self.camera_id)
        # 全链路延迟：小模型告警触发（本线程启动）→ 采样/判问 → 大模型判定返回
        t_end = time.monotonic()
        logger.info("vlm-review camera={} rule={} verdict={} frames={} "
                    "sample={}ms infer={}ms total={}ms persons={}",
                    self.camera_name, alert_type, verdict or "skipped",
                    len(crops) or len(persons),
                    round((t_sample - t0) * 1000), round((t_end - t_sample) * 1000),
                    round((t_end - t0) * 1000), len(persons))
        review = {"verdict": verdict or "skipped", "model": model,
                  "frames": len(crops) or len(persons)}
        if persons:
            review["persons"] = persons
        if not shadow and ((verdict is not None and verdict != expect)
                           or (verdict is None and not vlm.get("fail_open", True))):
            logger.info("vlm-review camera={} rule={} verdict={} → 丢弃",
                        self.camera_name, alert_type, review["verdict"])
            return
        self._build_and_send(rule_name, alert_type, snapshot, alert_objects,
                             iso_timestamp, review=None if shadow else review)

    # ------------------------------------------------------------------
    # 后台线程：JPEG 编码（executor）→ fire-and-forget 推送（daemon）
    # ------------------------------------------------------------------
    def _build_and_send(self, rule_name: str, alert_type: str, snapshot,
                        alert_objects: list[ObjectMeta], iso_timestamp: str,
                        review: dict | None = None):
        """JPEG 编码在 executor 线程执行（C 扩展，释放 GIL）。

        snapshot 是**未经 OSD 的原始帧**（证据支路挂在 nvdsosd 之前）；违规 red 框 +
        标签由 _render_evidence 用 payload 里的 ObjectMeta.bbox 现画，因此证据图
        是否带框与 nvdsosd / 预览 / 帧缓存滞后彻底无关，保证告警取证必有框。
        base64 / JSON / HTTP POST 交由独立 daemon 线程 fire-and-forget，
        executor 线程立即返回，不等待网络响应。
        """
        try:
            # 2. 补画违规框 + JPEG 编码（C 扩展，释放 GIL）— executor 线程唯一重操作；
            #    snapshot 为 None 时跳过，payload.frame_base64=null
            buffer = None
            if snapshot is not None and self.include_frame:
                # snapshot 是延迟句柄（EvidenceFrame）或 BGR ndarray；前者只在
                # 真正编码告警时触发 GPU→CPU，告警稀疏 → 常态零拷贝。
                frame_src = snapshot.bgr() if hasattr(snapshot, "bgr") else snapshot
                state = self._rule_states.get(rule_name)
                zones = state.rule.zones if state is not None else ()
                frame = self._render_evidence(frame_src, alert_type, alert_objects, zones)
                buffer = simplejpeg.encode_jpeg(frame, quality=85, colorspace='BGR')

            # 3. Fire-and-forget: base64 → JSON → HTTP 全部在独立 daemon 线程
            threading.Thread(
                target=self._send_payload,
                args=(rule_name, alert_type, buffer, alert_objects, iso_timestamp, review),
                daemon=True,
                name=f"webhook-send-{self.camera_id}-{alert_type}",
            ).start()

        except Exception:
            logger.exception("Webhook 构建异常")

    @staticmethod
    def _render_evidence(snapshot, alert_type: str, alert_objects: list[ObjectMeta],
                         zones: tuple[Zone, ...] = ()):
        """在原始证据帧上用 cv2 补画违规 person 红框 + 规则名标签（+ 区域轮廓）。

        返回**新数组**（不原地修改 FrameCache 持有的帧）。坐标为管线系 (x1,y1,x2,y2)，
        与 snapshot 同分辨率，直接对齐；越界自动裁剪。

        zones 非空时叠加区域轮廓（黄色闭合折线 + 区域 id）：入参是归一化坐标，按
        **本帧实际宽高**缩放，因此与 nvstreammux / 预览分辨率无关，换分辨率不用改配置。
        这一层是区域类告警的自证依据——图上能直接看出「人是否踩在区域内」。
        """
        frame = snapshot.copy()
        h, w = frame.shape[:2]
        color = (0, 0, 255)  # BGR 红

        # 先画区域（在框之下，避免遮挡违规框与标签）
        for zone in zones or ():
            # 映射与探针的 ZoneMatcher 一致：归一化 × 帧宽/高（不是 w-1），
            # 这样「图上看着在界内/界外」与「判定实际用的界」不会差出半个像素。
            pts = np.array([[int(min(max(0.0, x * w), w - 1)),
                             int(min(max(0.0, y * h), h - 1))]
                            for x, y in zone.points], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            tx, ty = int(pts[0][0]), max(14, int(pts[0][1]) - 6)
            cv2.putText(frame, zone.id, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA)

        for obj in alert_objects:
            x1, y1, x2, y2 = obj.bbox
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            (_, th), _ = cv2.getTextSize(alert_type, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ty = y1 - 8 if y1 - 8 - th >= 0 else min(h - 1, y2 + th + 8)
            cv2.putText(frame, alert_type, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return frame

    def _send_payload(self, rule_name: str, alert_type: str, buffer: bytes | None,
                      alert_objects: list[ObjectMeta], iso_timestamp: str,
                      review: dict | None = None):
        """Fire-and-forget: base64 → JSON → HTTP POST。独立 daemon 线程。

        GIL 密集操作（base64、json.dumps）和阻塞 I/O（HTTP）均在此线程完成，
        不阻塞 executor 线程或检测/预览管线。
        """
        try:
            # 1. Base64 编码（无证据帧时为 None）
            frame_base64 = None
            if buffer is not None:
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # 2. 构建告警 payload（alert_type = 对外报警类型，通常是基础算法名）
            objects_payload = []
            for obj in alert_objects:
                entry = {
                    "class": obj.class_name,
                    "confidence": round(obj.confidence, 3),
                    "bbox": list(obj.bbox),
                }
                attrs = [a for a in obj.attributes if a.name == rule_name]
                if attrs:
                    entry["attributes"] = [
                        {
                            "class": alert_type,
                            "confidence": round(attr.confidence, 3),
                            "bbox": list(attr.bbox) if attr.bbox is not None else None,
                            # 命中的区域 id（回指标定侧 RuleId）；全图规则为 null
                            "zone": attr.zone,
                        }
                        for attr in attrs
                    ]
                objects_payload.append(entry)

            # 本条告警命中的区域（配了区域的规则才有；供下游与标定侧闭环对齐）
            hit_zones = sorted({a.zone for obj in alert_objects for a in obj.attributes
                                if a.name == rule_name and a.zone})

            payload = {
                "event_id": f"evt_{uuid.uuid4().hex}",
                "schema_version": "1.0",
                "alert_type": alert_type,
                "camera_id": self.camera_id,
                "camera_name": self.camera_name,
                "device_code": self.device_code,
                "channel_no": self.channel_no,
                "timestamp": iso_timestamp,
                "zones": hit_zones,
                "objects": objects_payload,
                "frame_base64": frame_base64,
            }
            if review is not None:
                # VLM 复核结论（verdict/model/frames），下游可按需审计或展示
                payload["vlm"] = review

            # 3. HTTP POST（阻塞 I/O，释放 GIL）
            self.webhook.send(payload)

        except Exception:
            logger.exception("Webhook 推送异常")
