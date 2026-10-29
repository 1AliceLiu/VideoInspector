#!/usr/bin/env python3
"""配置加载与校验（deploy/config.yaml）。"""

import yaml
from pathlib import Path

from server.alert.rules import parse_rules, TRIGGER_PRESENCE
from server.model_spec import parse_gies, KIND_DETECTOR


def load_config(config_path: str) -> dict:
    """加载并校验 YAML 配置文件。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config, path)
    return config


def validate_config(config: dict, path: Path):
    """校验: model.gies / rules / 摄像头(active_rules) / webhook / source.reconnect。"""
    errors = []

    model = config.get("model") or {}
    gies: dict = {}
    gies_raw = model.get("gies")
    if not gies_raw:
        errors.append("model.gies 为必填项（至少声明一个锚点检测器）")
    else:
        try:
            gies = parse_gies(gies_raw)
        except ValueError as exc:
            errors.append(str(exc))

    person_conf = model.get("person_conf_threshold")
    if person_conf is not None and not (
        isinstance(person_conf, (int, float)) and 0.0 <= person_conf <= 1.0
    ):
        errors.append(f"model.person_conf_threshold 必须是 0~1 的置信度阈值，当前: {person_conf!r}")

    rules_raw = config.get("rules")
    rules: dict = {}
    rule_names: set[str] = set()
    if not rules_raw:
        errors.append("缺少 'rules' 节（至少定义一个告警规则）")
    else:
        try:
            rules = parse_rules(rules_raw)
            rule_names = set(rules)
            for rule in rules.values():
                spec = gies.get(rule.gie)
                if spec is None:
                    errors.append(f"rules.{rule.name}.gie 引用了未定义的模型: {rule.gie}")
                    continue
                # presence 型（区域入侵）不检违规类，只认锚点检出 → 必须挂在无
                # violation 的锚点检测器上；挂在报警模型上会让「违规」与「在场」混淆
                if rule.trigger == TRIGGER_PRESENCE:
                    if spec.kind != KIND_DETECTOR or spec.violation is not None:
                        errors.append(
                            f"rules.{rule.name}.gie={rule.gie} 必须是锚点检测器"
                            "（kind=detector 且无 violation）才能用作 presence 判定"
                        )
        except ValueError as exc:
            errors.append(str(exc))

    cameras = [c for c in config.get("cameras") or [] if c.get("enabled", True)]
    if not cameras:
        errors.append("cameras 必须是非空列表（至少一个 enabled=true）")
    else:
        generated_prefix = "__cam__"      # 与 api.config_store.GENERATED_PREFIX 一致
        for cam in cameras:
            if cam.get("type", "rtsp") != "rtsp":
                errors.append(f"DeepStream 仅支持 RTSP 摄像头: {cam.get('id')} (type={cam.get('type')})")
            if not cam.get("rtsp_url"):
                errors.append(f"cameras[].rtsp_url 为必填项 (type=rtsp): {cam.get('id')}")
            active = cam.get("active_rules")
            if active is not None:
                if not isinstance(active, list) or not active:
                    errors.append(f"cameras[{cam.get('id')}].active_rules 必须是非空列表")
                else:
                    for rule in active:
                        if rule not in rule_names:
                            errors.append(f"cameras[{cam.get('id')}].active_rules 引用了未定义的规则: {rule}")
            else:
                # 未声明 = 该路激活所有非生成态规则（与 builder._active_rules_by_source 一致）
                active = [n for n in rule_names if not n.startswith(generated_prefix)]

            # presence 型规则的实际生效版本必须带区域：基础规则可作无 zones 的模板
            # （区域由按摄像头生成的 __cam__ 规则补上），但「某一路跑 presence 且无区域」
            # 等于「画面里出现任何人都是入侵」，必然刷屏 → 在启动期拒绝。
            for name in active:
                rule = rules.get(name)
                if rule is not None and rule.trigger == TRIGGER_PRESENCE and not rule.zones:
                    errors.append(
                        f"cameras[{cam.get('id')}] 生效的规则 {name} 是 presence 型（区域入侵）"
                        f"但未绑定 zones —— 任何位置出现人员都会判为入侵；"
                        "请为该摄像头配置区域，或从 active_rules 中移除该规则"
                    )

    # VLM 复核（吸烟等告警二次确认，见 alert/vlm_review.py）：开启时 url 必填，
    # rules 必须引用已定义规则，防止启动后复核门静默失效。
    vlm_cfg = config.get("vlm") or {}
    if vlm_cfg.get("enabled"):
        if not vlm_cfg.get("url"):
            errors.append("vlm.enabled=true 时 vlm.url 为必填项（OpenAI 兼容 /v1/chat/completions）")
        unknown = [r for r in vlm_cfg.get("rules") or [] if r not in rule_names]
        if unknown:
            errors.append(f"vlm.rules 引用了未定义的规则: {unknown}")

    webhook = config.get("webhook") or {}
    timeout = webhook.get("timeout")
    if webhook.get("url") and timeout is not None and not (
        isinstance(timeout, (int, float)) and timeout > 0
    ):
        errors.append(f"webhook.timeout 必须是正数，当前: {timeout!r}")

    errors.extend(_validate_source_reconnect((config.get("source") or {}).get("reconnect")))
    errors.extend(_validate_source_health(((config.get("source") or {}).get("health")) or {}))

    drop = (config.get("source") or {}).get("drop_frame_interval")
    if drop is not None and (not isinstance(drop, int) or isinstance(drop, bool) or drop < 1):
        errors.append(
            f"source.drop_frame_interval 必须是不小于 1 的整数（1=不抽帧，N=每 N 帧放行 1 帧），"
            f"当前: {drop!r}"
        )

    if errors:
        raise ValueError(f"配置校验失败 ({path}):\n  " + "\n  ".join(errors))


def _validate_int(v, name: str, minimum: int) -> str | None:
    if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
        return f"{name} 必须是不小于 {minimum} 的整数，当前: {v!r}"
    return None


def _validate_source_reconnect(rc_raw) -> list[str]:
    errors = []
    rc = rc_raw or {}
    checks = (
        ("attempts", 1),
        ("timeout", 1),
        ("interval", 1),
        ("latency", 0),
        ("stall_seconds", 60),
    )
    for key, minimum in checks:
        v = rc.get(key)
        if v is not None and (err := _validate_int(v, f"source.reconnect.{key}", minimum)):
            errors.append(err)
    return errors


def _validate_source_health(h_raw) -> list[str]:
    errors = []
    h = h_raw or {}
    checks = (
        ("source_stall_seconds", 60),
        ("check_interval", 10),
        ("min_uptime_seconds", 0),
        ("rebuild_cost_seconds", 60),
        ("min_rebuild_interval", 60),
    )
    for key, minimum in checks:
        v = h.get(key)
        if v is not None and (err := _validate_int(v, f"source.health.{key}", minimum)):
            errors.append(err)
    window = h.get("maintenance_window")
    if window is not None:
        try:
            hh, mm = (int(x) for x in str(window).split(":"))
            if not (0 <= hh < 24 and 0 <= mm < 60):
                raise ValueError
        except ValueError:
            errors.append(f"source.health.maintenance_window 必须是 HH:MM（UTC），当前: {window!r}")
    return errors


def resolve_path(base: Path, p: str) -> str:
    """把配置里的相对路径解析为绝对路径（相对部署根 = config.yaml 所在目录）。"""
    return p if Path(p).is_absolute() else str(base / p)


# 管线子进程因「断流卡死被看门狗判定不可恢复」时的退出码。
# 主进程监督循环据此决定是否重建整条管线。
EXIT_PIPELINE_STUCK = 2

# 看门狗默认参数（可由 source.reconnect 覆盖）
DEFAULT_RECONNECT = {
    # nvurisrcbin 内置重连：消耗完 attempts 后该路源进入僵死状态，
    # 由看门狗检测并退出子进程、监督循环全量重建管线（见 server/watchdog.py）。
    "attempts": 10,       # 内置重连次数上限（有限值，不做无限空转；有外层兜底可适当放大）
    "timeout": 10,        # 多少秒无数据包即判为断流（5s 偏激进易误判）
    "interval": 10,       # 重连尝试间隔（秒）
    "latency": 500,       # jitterbuffer 缓冲（ms），过大延迟超时判定
    "stall_seconds": 180, # 连续多久没有任何健康帧心跳则认为管线卡死
}

# 逐路健康监控默认参数（可由 source.health 覆盖，见 server/pipeline/health.py）
# 决策 = 带切换成本的在线区间覆盖: min(K·|R| + α·Σ UncoveredTime)
DEFAULT_HEALTH = {
    "source_stall_seconds": 300,     # 该路超过 N 秒无帧 → stalled（未覆盖起点）
    "check_interval": 30,            # 巡检周期（秒）
    "min_uptime_seconds": 120,       # 子进程启动预热期，期内不判定
    # 代价阈值 K（未覆盖相机秒）: W=Σ未覆盖时长 ≥ K 且距上次重建足够久 → 重建。
    # 单路断约 15min 攒够，两路 7.5min——多路断自动加速，即 batching 防抖。
    "rebuild_cost_seconds": 900,
    # 防风暴硬地板: 两次重建最小间隔。重建后相机仍离线时 W 重新累积，
    # 但最快也只按此间隔重试"相机已回归但 DS 已放弃"的场景。
    "min_rebuild_interval": 1800,
    # 网关每日定时重启的 UTC 时刻（北京时间 00:04 = UTC 16:04）±5min 内只
    # 积累 W 不动手，窗口结束后一并结算——计划内重启顺延收编窗口期上线相机
    "maintenance_window": "16:04",
}
