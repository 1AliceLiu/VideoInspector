"""
告警规则配置模型（从 YAML rules 节解析）

每个告警规则（no_helmet / no_vest / no_harness / ...）独立持有策略:
  - gie: 引用的 model.gies 名称（该 gie 的 violation 即报警类）
  - cooldown_seconds:    触发告警后的冷却时长（同一规则同一摄像头）
  - min_detection_count: 连续检测到违规的帧数阈值（防单帧误报）
  - attribute_threshold: 违规置信度门限；detector 作空间关联门限，
                         classifier 重写进 INI 的 classifier-threshold
  - trigger: violation(默认，命中违规属性才告警) | presence(区域内出现即告警)
  - zones:   本规则的空间作用域（归一化多边形，见 alert.regions）。
             **省略 = 全图生效**；presence 型规则必须绑定区域（否则退化成
             「画面里有任何人 = 入侵」，启动期直接拒绝）
  - vacant_frames: presence 型的「离开去抖」——区域内连续 N 帧无人才认定事件
             结束并复位，等待下一次进入（防遮挡/贴边抖动被误判成人已离开）

规则之间相互独立，各自维护状态机，互不竞争。
`alert_type` 是 webhook 对外报警类型，未配置时等于规则名；按摄像头生成的
`__cam__` 规则用它回指基础算法，避免 payload 泄漏内部规则名。
"""

from __future__ import annotations

from dataclasses import dataclass

from server.alert.regions import parse_zones

TRIGGER_VIOLATION = "violation"   # 违规属性命中（现有 PPE 类算法）
TRIGGER_PRESENCE = "presence"     # 区域内出现目标即命中（区域入侵）
KNOWN_TRIGGERS = (TRIGGER_VIOLATION, TRIGGER_PRESENCE)


@dataclass(frozen=True)
class RuleConfig:
    """单条告警规则的独立策略。"""

    name: str
    gie: str                        # 引用的 model.gies 名称
    cooldown_seconds: float
    min_detection_count: int
    attribute_threshold: float
    # OSD/payload 展示名（可选，预留字段）：未配置时回退为规则名。
    # 当前 OSD 与 payload 仍使用规则名（ASCII），渲染行为暂不改变。
    display_name: str | None = None
    # 对外的报警类型（webhook alert_type）；未配置时等于规则名。
    # 按摄像头生成的 __cam__ 规则用它回指基础算法，避免 payload 泄漏内部规则名。
    alert_type: str | None = None
    # 命中语义：violation=检出违规属性；presence=区域内出现锚点目标（区域入侵）
    trigger: str = TRIGGER_VIOLATION
    # 空间作用域：空元组 = 全图；非空 = 仅区域内命中计入本规则
    zones: tuple = ()
    # presence 型的离开去抖帧数（事件复位阈值），violation 型忽略
    vacant_frames: int = 1


def parse_rules(raw: dict | None) -> dict[str, RuleConfig]:
    """把 YAML rules 节解析成 {rule_name: RuleConfig}。

    Raises:
        ValueError: 参数非法（启动时 fail fast）。
    """
    rules: dict[str, RuleConfig] = {}
    for name, cfg in (raw or {}).items():
        cfg = cfg or {}
        gie = cfg.get("gie")
        cooldown = cfg.get("cooldown_seconds")
        min_count = cfg.get("min_detection_count")
        threshold = cfg.get("attribute_threshold")
        display_name = cfg.get("display_name")
        alert_type = cfg.get("alert_type")
        trigger = cfg.get("trigger", TRIGGER_VIOLATION)
        vacant_frames = cfg.get("vacant_frames", 1)
        if not isinstance(gie, str) or not gie:
            raise ValueError(f"rules.{name}.gie 为必填项（引用 model.gies 名称）")
        if not isinstance(cooldown, (int, float)) or cooldown < 0:
            raise ValueError(f"rules.{name}.cooldown_seconds 必须是非负数")
        if not isinstance(min_count, int) or min_count < 1:
            raise ValueError(f"rules.{name}.min_detection_count 必须是 >=1 的整数")
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            raise ValueError(f"rules.{name}.attribute_threshold 必须是 0~1 的置信度")
        if display_name is not None and not isinstance(display_name, str):
            raise ValueError(f"rules.{name}.display_name 必须是字符串")
        if alert_type is not None and (not isinstance(alert_type, str) or not alert_type):
            raise ValueError(f"rules.{name}.alert_type 必须是非空字符串")
        if trigger not in KNOWN_TRIGGERS:
            raise ValueError(f"rules.{name}.trigger 必须是 {KNOWN_TRIGGERS}，当前: {trigger!r}")
        if not isinstance(vacant_frames, int) or vacant_frames < 1:
            raise ValueError(f"rules.{name}.vacant_frames 必须是 >=1 的整数")
        zones = parse_zones(cfg.get("zones"), where=f"rules.{name}")
        # 注：presence 型「必须有区域」不在这里判定 —— 基础规则常作为模板声明
        # （不带 zones，由按摄像头生成的 __cam__ 规则补上），真正的约束在
        # config.validate_config 按「摄像头实际生效的规则」校验，避免模板无法声明。
        rules[name] = RuleConfig(
            name=name,
            gie=gie,
            cooldown_seconds=float(cooldown),
            min_detection_count=int(min_count),
            attribute_threshold=float(threshold),
            display_name=(display_name or name),
            alert_type=(alert_type or name),
            trigger=trigger,
            zones=zones,
            vacant_frames=int(vacant_frames),
        )
    return rules
