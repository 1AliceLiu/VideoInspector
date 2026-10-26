"""
区域（zone）模型与几何判定 —— 规则的空间作用域

语义（单一实现，配置层与探针层共用）:
  - 一条规则可绑定 0..N 个区域；**未绑定区域 = 全图生效**（现有算法的默认形态）。
  - 区域用自由多边形表达（≥3 顶点），坐标一律为**归一化 0~1**（相对画面宽高）。
    归一化的理由不是省事，而是坐标系无关：nvstreammux 默认拉伸填满 mux 尺寸
    （未启用 enable-padding），拉伸对归一化坐标是不变的
    （x_pix = X × mux_w，y_pix = Y × mux_h），因此平台在原始视频上标定的点
    可直接与检测框比对，换相机分辨率/宽高比无需重标。
    ⚠ 若将来给 mux 打开 enable-padding（letterbox），此推导失效，需扣除 padding 偏移。
  - 多边形**不做任何简化**（不共线压缩、不 Douglas-Peucker）：贴画面边框的顶点是
    标定者表达「区域延伸出画面外」的唯一手段，删掉会改变区域形状。

锚点（anchor）—— 一个像素点代表「这个人站在哪」:
  - feet   底边中点 (cx, y2)：人「踩进」区域，与地面型多边形（透视梯形）匹配，
           远景小框最稳。默认。
  - center 框中心：适合窄区域/俯拍。
"""

from __future__ import annotations

from dataclasses import dataclass

ANCHOR_FEET = "feet"
ANCHOR_CENTER = "center"
KNOWN_ANCHORS = (ANCHOR_FEET, ANCHOR_CENTER)

MIN_POINTS = 3
MAX_POINTS = 32   # 顶点数直接进每帧判定循环，异常点云在启动期就拦掉


@dataclass(frozen=True)
class Zone:
    """单个区域（归一化多边形）。id 用于把告警回指到标定侧的规则号。"""

    id: str
    points: tuple[tuple[float, float], ...]
    anchor: str = ANCHOR_FEET


def parse_zones(raw, *, where: str) -> tuple[Zone, ...]:
    """解析并校验 zones 节 → 不可变 Zone 元组。空值表示「不限定区域」（全图）。

    Args:
        raw: YAML/JSON 里的 zones 列表，每项 {id?, anchor?, points: [[x,y], ...]}；
             也允许直接给一个多边形（裸 points 列表），便于手写配置。
        where: 出错信息里的定位前缀，如 "rules.no_intrusion"。

    Raises:
        ValueError: 结构非法 / 点数越界 / 坐标不在 0~1 / 顶点不足 / 边自交。
    """
    if raw is None:
        return ()
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where}.zones 必须是非空列表（或省略表示全图）")

    zones: list[Zone] = []
    for i, item in enumerate(raw):
        at = f"{where}.zones[{i}]"
        if isinstance(item, (list, tuple)):      # 裸多边形 [[x,y],...]
            item = {"points": item}
        if not isinstance(item, dict):
            raise ValueError(f"{at} 必须是对象或 points 列表")

        # 同时接受本方命名(points)与平台命名(Points)，便于直接贴平台 JSON
        pts = _parse_points(item.get("points", item.get("Points")), at)
        if _self_intersects(pts):
            raise ValueError(f"{at} 多边形边自交（非法区域，请检查顶点顺序）")

        anchor = item.get("anchor", ANCHOR_FEET)
        if anchor not in KNOWN_ANCHORS:
            raise ValueError(f"{at}.anchor 必须是 {KNOWN_ANCHORS}，当前: {anchor!r}")

        zid = item.get("id") or item.get("RuleId") or f"z{i + 1}"
        if not isinstance(zid, str) or not zid:
            raise ValueError(f"{at}.id 必须是非空字符串")
        zones.append(Zone(id=zid, points=pts, anchor=anchor))

    ids = [z.id for z in zones]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{where}.zones 的 id 不能重复: {ids}")
    return tuple(zones)


def _parse_points(raw, at: str) -> tuple[tuple[float, float], ...]:
    """[[x,y],...] → 去掉连续重复点后的归一化顶点元组（保留贴边顶点，不简化）。"""
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"{at}.points 必须是非空的 [[x, y], ...] 列表")

    pts: list[tuple[float, float]] = []
    for j, p in enumerate(raw):
        # 同时接受 [[x,y],...] 与 平台侧 [{"X":..,"Y":..},...] 两种形态
        if isinstance(p, dict):
            xy = (p.get("X", p.get("x")), p.get("Y", p.get("y")))
        else:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError(f"{at}.points[{j}] 必须是 [x, y] 两个数")
            xy = (p[0], p[1])
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in xy):
            raise ValueError(f"{at}.points[{j}] 坐标必须是 0~1 的数值，当前: {p!r}")
        if not all(0.0 <= float(v) <= 1.0 for v in xy):
            raise ValueError(f"{at}.points[{j}] 坐标必须是归一化 0~1，当前: {p!r}")

        pt = (float(xy[0]), float(xy[1]))
        if pts and pts[-1] == pt:        # 仅去连续重复（闭合重复首点属此类）
            continue
        pts.append(pt)

    if pts and pts[0] == pts[-1] and len(pts) > MIN_POINTS:
        pts.pop()          # 显式闭合的首尾重复点 → 去掉（否则构成零长度边）

    if len(pts) > MAX_POINTS:
        raise ValueError(f"{at}.points 顶点数 {len(pts)} 超过上限 {MAX_POINTS}")
    if len(pts) < MIN_POINTS:
        raise ValueError(f"{at}.points 去重后只剩 {len(pts)} 个顶点，至少需要 {MIN_POINTS} 个")
    return tuple(pts)


def _self_intersects(pts: tuple[tuple[float, float], ...]) -> bool:
    """任意两条**不相邻**的边相交即为非法简单多边形（启动期检查，O(n²) 可接受）。"""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            if i == j or (j == i + 1) or (i == 0 and j == n - 1):
                continue
            if _segments_cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n]):
                return True
    return False


def _cross(ox, oy, px, py, qx, qy) -> float:
    return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)


def _segments_cross(a, b, c, d) -> bool:
    d1, d2 = _cross(*a, *b, *c), _cross(*a, *b, *d)
    d3, d4 = _cross(*c, *d, *a), _cross(*c, *d, *b)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


class ZoneMatcher:
    """归一化区域 → 像素空间的命中器（启动期预换算，热路径只做比较）。

    用法: 探针每帧为每人算出 feet/center 两个代表点后调用 hit(feet, center)
    → 命中的 Zone 或 None。每个区域按自己的 `anchor` 取对应的那个点判定，
    因此同一帧只算一次点、可安全服务混配锚点的多条区域规则。
    加速: 先比多边形外接矩形（绝大多数帧在此淘汰），命中才进射线法。
    """

    def __init__(self, zones: tuple[Zone, ...], width: int, height: int):
        self._entries = []
        for z in zones or ():
            pix = tuple((x * width, y * height) for x, y in z.points)
            xs = [p[0] for p in pix]
            ys = [p[1] for p in pix]
            self._entries.append((z, pix, min(xs), max(xs), min(ys), max(ys)))

    def __bool__(self) -> bool:
        return bool(self._entries)

    def hit(self, feet: tuple[float, float],
            center: tuple[float, float]) -> Zone | None:
        """返回包含该人员的区域（多区域命中取配置中先出现者）。"""
        for zone, pix, x1, x2, y1, y2 in self._entries:
            px, py = center if zone.anchor == ANCHOR_CENTER else feet
            if not (x1 <= px <= x2 and y1 <= py <= y2):     # 外接矩形粗筛
                continue
            if point_in_polygon(px, py, pix):
                return zone
        return None


def point_in_polygon(x: float, y: float, pts: tuple[tuple[float, float], ...]) -> bool:
    """射线法（奇偶穿越）：点是否在多边形内。边界点视为内部（与标定侧不敏感）。"""
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def anchor_point(x1: int, y1: int, x2: int, y2: int, anchor: str) -> tuple[float, float]:
    """检测框 (x1,y1,x2,y2) → 该锚点语义下的代表点。"""
    cx = (x1 + x2) / 2.0
    if anchor == ANCHOR_CENTER:
        return cx, (y1 + y2) / 2.0
    return cx, float(y2)                                   # feet：底边中点
