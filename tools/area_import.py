#!/usr/bin/env python3
"""
平台区域标定文件 → 配置导入器

用途: 把上游平台导出的区域 JSON（形如 2628-area.txt: RuleId / RuleType / Points /
Algo{majorId,minorId,name}）写进 config.yaml 的指定摄像头，绑定到对应算法。

为什么走 ConfigStore 而不是直接编辑 YAML:
  ConfigStore 自带 备份 → 校验(_validate_candidate) → 写盘 → revision 递增 →
  reload 请求 的完整链路，改坏了可一键 rollback；手改文本没有这层保护。

算法绑定关系（本平台按「每算法各配自己的区域」建模）:
  区域挂在「(摄像头, 算法) 的规则实例」上，即 API 的 __cam__ 生成规则，
  与平台侧 RuleId 一一对应。同一算法的多条 RuleId 聚合为该规则的 zones 列表。

坐标: 平台已是归一化 0~1（相对画面宽高），直接落库不做换算。理由见
  server/alert/regions.py —— nvstreammux 拉伸填满，归一化对拉伸不变。

用法:
    python tools/area_import.py --area 2628-area.txt --camera 2628 \
        --config deploy/config.yaml [--dry-run] [--rule-map 9=intrusion 4=no_helmet]

    # 之后确认无误再重启/重载使检测进程生效（本工具只改配置，不重启进程）

未知 minorId **直接报错要求人工补表**，绝不猜测——我们没有平台的算法字典。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.alert.regions import parse_zones          # noqa: E402
from server.api.config_store import ConfigStore       # noqa: E402
from server.api.errors import ApiError                # noqa: E402

# 平台算法编号 → 本项目规则名。默认表只含已确认的项；其余用 --rule-map 补充。
# 已确认: 2628-area.txt 的 Algo.minorId=9 / name="区域入侵" → intrusion
DEFAULT_RULE_MAP = {9: "intrusion"}

# 平台的区域点已是归一化坐标；导入时保留的小数位（与源文件精度一致）
COORD_PRECISION = 4


def load_area(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{path}: 顶层必须是区域列表")
    return data


def zone_of(entry: dict) -> dict:
    """平台一条区域 → 我们的 zone 字典（校验在 parse_zones 里做，保持单一实现）。"""
    pts = [
        [round(float(p["X"]), COORD_PRECISION), round(float(p["Y"]), COORD_PRECISION)]
        for p in entry["Points"]
    ]
    return {"id": str(entry.get("RuleId") or ""), "anchor": "feet", "points": pts}


def resolve_rule(entry: dict, rule_map: dict[int, str]) -> str:
    algo = entry.get("Algo") or {}
    minor = algo.get("minorId")
    if minor not in rule_map:
        raise SystemExit(
            f"未知算法编号 minorId={minor} (name={algo.get('name')!r}, "
            f"RuleId={entry.get('RuleId')!r})\n"
            f"  当前映射: {rule_map}\n"
            f"  请用 --rule-map {minor}=<规则名> 指定（规则名须已存在于 config.rules）；"
            f"本工具不猜映射。"
        )
    return rule_map[minor]


def build_zones(entries: list[dict], rule_map: dict[int, str]) -> dict[str, list[dict]]:
    """平台区域列表 → {规则名: [zone, ...]}（同算法多区域自动聚合）。"""
    out: dict[str, list[dict]] = {}
    for entry in entries:
        if not entry.get("Points"):
            raise SystemExit(f"RuleId={entry.get('RuleId')!r} 缺少 Points")
        rule = resolve_rule(entry, rule_map)
        zone = zone_of(entry)
        # 与运行期同一套校验：坏区域在导入期就拦住，不留到启动
        parse_zones([zone], where=f"rules.{rule}")
        zones = out.setdefault(rule, [])
        if any(z["id"] == zone["id"] for z in zones):
            raise SystemExit(f"规则 {rule} 下区域 id 重复: {zone['id']}")
        zones.append(zone)
    return out


def plan(store: ConfigStore, camera_id: str, zones_by_rule: dict[str, list]) -> list[dict]:
    """目标 algorithms 列表：保留该路既有算法，仅叠加区域。

    用公开视图 list_cameras() 取现状（{id, params} 形状可直接回灌 set_algorithms），
    不碰 ConfigStore 内部结构，也不重复实现 __cam__ 的解析逻辑。

    只下发**与基础规则取值不同**的参数：参数全等时 set_algorithms 会走「不生成
    __cam__ 规则、直接引用基础规则」的分支，从而避免仅为导入区域就把该路其它
    算法的形状改成生成态（配置 diff 噪声 = 排障成本）。
    """
    cams = {str(c["id"]): c for c in store.list_cameras()}
    cam = cams.get(str(camera_id))
    if cam is None:
        raise SystemExit(f"摄像头不存在: {camera_id}")

    base_rules = store.read().get("rules") or {}
    current = []
    for alg in cam.get("algorithms") or []:
        base = base_rules.get(alg["id"]) or {}
        diff = {k: v for k, v in (alg.get("params") or {}).items()
                if not (k in base and base[k] == v)}
        current.append({"id": alg["id"], "params": diff})

    have = {a["id"] for a in current}
    for rule, zones in zones_by_rule.items():
        if rule not in base_rules:
            raise SystemExit(
                f"规则 {rule!r} 不在 config.rules 中；请先在 rules 节声明"
                f"（区域入侵形如: {rule}: {{gie: person, trigger: presence, ...}}）"
            )
        if rule in have:
            for item in current:
                if item["id"] == rule:
                    item["params"]["zones"] = zones
        else:
            current.append({"id": rule, "params": {"zones": zones}})
    return current


def main() -> int:
    ap = argparse.ArgumentParser(description="平台区域标定文件导入器")
    ap.add_argument("--area", required=True, type=Path, help="平台导出的区域 JSON")
    ap.add_argument("--camera", required=True, help="目标摄像头 id")
    ap.add_argument("--config", default="deploy/config.yaml", help="config.yaml 路径")
    ap.add_argument("--runtime-dir", default=None,
                    help="runtime 目录（默认 config 同目录下的 runtime/）")
    ap.add_argument("--rule-map", action="append", default=[], metavar="MINOR=RULE",
                    help="补充/覆盖 minorId→规则名 映射，可多次")
    ap.add_argument("--dry-run", action="store_true", help="只打印将写入的内容")
    args = ap.parse_args()

    rule_map = dict(DEFAULT_RULE_MAP)
    for item in args.rule_map:
        try:
            minor, rule = item.split("=", 1)
            rule_map[int(minor)] = rule.strip()
        except ValueError:
            raise SystemExit(f"--rule-map 形如 9=intrusion，当前: {item!r}")

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"配置不存在: {cfg_path}")
    runtime = Path(args.runtime_dir) if args.runtime_dir else cfg_path.parent / "runtime"

    entries = load_area(args.area)
    zones_by_rule = build_zones(entries, rule_map)

    store = ConfigStore(cfg_path, runtime)
    algorithms = plan(store, args.camera, zones_by_rule)

    print(f"区域文件: {args.area}  条目: {len(entries)}")
    for rule, zones in zones_by_rule.items():
        for z in zones:
            print(f"  → {rule}  zone={z['id']}  顶点={len(z['points'])}  anchor={z['anchor']}")
    print(f"写入后 {args.camera} 的 algorithms: {json.dumps(algorithms, ensure_ascii=False)}")

    if args.dry_run:
        print("\n[dry-run] 未写入配置")
        return 0

    # set_algorithms 只改内存中的候选配置（校验通过才返回），落盘由 activate 完成
    before = copy.deepcopy(store.read())
    try:
        candidate = store.set_algorithms(args.camera, algorithms)
    except ApiError as exc:
        print(f"\n写入被拒绝（配置未变更）: {exc.code} {exc.message}", file=sys.stderr)
        return 2

    if candidate == before:
        print("配置无变化，跳过 activate")
        return 0

    revision = store.activate(candidate)
    print(f"\n✓ 已写入并请求重载 (revision={revision})")
    print("  注意: 检测进程按重载/重启后才生效；本工具不重启进程。")
    print(f"  回滚: 备份在 {store.runtime_dir / 'config-history'}，或 "
          f"ConfigStore.rollback(<备份文件>) / 重新导入旧的标定文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
