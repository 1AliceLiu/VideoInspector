#!/usr/bin/env python3
"""
区域轮廓预览（标定期核对，不触发告警、不进检测管线）

为什么需要它: 运行期的区域轮廓只在**告警成单**时才会画到证据图上
（server/alert/manager._render_evidence）。若测试时段区域内没人走过，就拿不到图，
无法确认「标定多边形到底贴不贴画面」。本工具抓一帧（或吃一张现成快照）直接把区域
叠上去，几秒完成，与检测进程完全无关。

关键性质: 区域是归一化 0~1 坐标，按**帧实际宽高**线性缩放即可对齐 —— 因为
nvstreammux 未启用 enable-padding（拉伸填满 1920x1080），拉伸对归一化坐标不变。
所以本工具在相机原始分辨率上画的图，与检测器在 mux 尺寸下看到的相对位置一致。

取帧方式（与项目既有 RTSP 排障手段同源，见 server/api/probe.py）:
    ffmpeg -rtsp_transport tcp -i <url> -frames:v 1   →   失败回退 cv2.VideoCapture

⚠ 拉流会为该 URI 增加一路**并发订阅**（生产实例通常已在拉同一路）。只读一帧、
   秒级退出，负载很小；但若网关并发超限而踢掉既有会话，受影响的是**这一路**。
   完全不想碰流：用 --image 传一张平台给的快照，或用 --stats-only 只看几何。

用法:
    # 影子实例配置（区域内联在 rules 上）
    python3 tools/preview_zone.py --config deploy/config.yaml --camera 2628

    # 生产配置（区域由 tools/area_import.py 生成的 __cam__ 规则携带，同样能读）
    python3 tools/preview_zone.py --config deploy/config.yaml --camera 2628

    # 导入前先看平台标定文件画出来是什么样
    python3 tools/preview_zone.py --area 2628-area.txt --image snapshot-2628.jpg

    # 不碰流、也不画图，只核对几何
    python3 tools/preview_zone.py --config deploy/config.yaml --camera 2628 --stats-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.alert.regions import parse_zones          # noqa: E402
from server.api.util import mask_rtsp_url             # noqa: E402
from server.config import load_config                 # noqa: E402

GENERATED_PREFIX = "__cam__"

# 每条规则一个颜色，便于同画面上区分「入侵的区域」与「作业面的区域」
PALETTE = [
    (0, 255, 255),    # 黄
    (255, 128, 0),    # 橙蓝
    (0, 128, 255),    # 青红
    (255, 0, 255),    # 品红
    (128, 255, 0),    # 黄绿
]


# ------------------------------------------------------------------ 取帧
def grab_frame(url: str, timeout: float, frames: int) -> np.ndarray | None:
    """抓一帧：优先 ffmpeg（RTSP 兼容性最好），失败回退 cv2。"""
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "frame.png"
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                   "-rtsp_transport", "tcp", "-i", url,
                   "-frames:v", "1", "-y", str(out)]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"ffmpeg 取帧超时（{timeout}s）", file=sys.stderr)
            else:
                if proc.returncode == 0 and out.exists():
                    img = cv2.imread(str(out))
                    if img is not None:
                        return img
                    print("ffmpeg 出图但无法解码", file=sys.stderr)
                else:
                    err = (proc.stderr or "").strip().splitlines()
                    print(f"ffmpeg 取帧失败: {err[-1] if err else proc.returncode}",
                          file=sys.stderr)

    cap = cv2.VideoCapture(url)
    try:
        if not cap.isOpened():
            return None
        # 跳过开头若干帧：RTSP 首帧常被截/无 SPS-PPS，颜色会花
        best = None
        for _ in range(max(1, frames)):
            ok, img = cap.read()
            if ok and img is not None:
                best = img
            else:
                break
        return best
    finally:
        cap.release()


# ------------------------------------------------------------------ 几何核对
def _area_ratio(points) -> float:
    """鞋带公式：多边形面积占画面比例（归一化空间）。"""
    n = len(points)
    a = 0.5 * sum(points[i][0] * points[(i + 1) % n][1]
                  - points[(i + 1) % n][0] * points[i][1] for i in range(n))
    return abs(a)


def _deflections(points):
    """各顶点偏转角（度）。0=直连（冗余点），~90 且贴边框=画面角产物。"""
    import math
    out = []
    n = len(points)
    for i in range(n):
        p0, p1, p2 = points[i - 1], points[i], points[(i + 1) % n]
        v1 = (p0[0] - p1[0], p0[1] - p1[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        ang = math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0],
                                      v1[0] * v2[0] + v1[1] * v2[1]))
        out.append(180 - abs(ang))
    return out


BORDER_TOL = 0.02    # 归一化容差：1080p 下约 21px，小于此值视为「贴画面边框」


def describe_zone(zone, ref_height: int = 1080, ref_width: int = 1920) -> list[str]:
    """把几何事实翻成人在标定要检查的话——异常不自检，靠这里提示。"""
    notes: list[str] = []
    pts = zone.points
    ratio = _area_ratio(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    print(f"    {zone.id}  顶点={len(pts)}  anchor={zone.anchor}  "
          f"面积占画面={ratio * 100:.1f}%  外接框={bw:.2f}×{bh:.2f}")

    def edges(x, y):
        hit = []
        if x < BORDER_TOL:
            hit.append("左")
        if x > 1 - BORDER_TOL:
            hit.append("右")
        if y < BORDER_TOL:
            hit.append("上")
        if y > 1 - BORDER_TOL:
            hit.append("下")
        return hit

    degs = _deflections(pts)

    # 同时贴两条边 + 偏转≈90° → 这个顶点就是画面角本身，是标定者闭合多边形的
    # 产物（禁区在该方向延伸出画面），不是物理拐点。删掉它会改变区域形状。
    for i, (x, y) in enumerate(pts):
        e = edges(x, y)
        if len(e) >= 2 and abs(degs[i] - 90) < 8:
            notes.append(
                f"V{i + 1} ({x:.4f},{y:.4f}) 同时贴「{'/'.join(e)}」两边且偏转 "
                f"{degs[i]:.1f}° → 这是画面角本身，禁区在该方向延伸出画面，"
                f"非物理拐点（不可删，删了区域会缺一角）")

    touched = [e for e in ("左", "右", "上", "下")
               if any(e in edges(x, y) for x, y in pts)]
    if touched:
        n = sum(1 for x, y in pts if edges(x, y))
        # 各轴按自己的像素尺度算最近距离，别把横向距离用帧高换算
        dists = {
            "左": min(xs) * ref_width,
            "右": (1 - max(xs)) * ref_width,
            "上": min(ys) * ref_height,
            "下": (1 - max(ys)) * ref_height,
        }
        detail = "、".join(f"{e}边≈{dists[e]:.0f}px" for e in touched)
        notes.append(
            f"{n} 个顶点贴画面边框（{detail}）"
            f"→ 该方向禁区被画面切断，**画面最边缘的人员几乎必然命中**；"
            f"请确认是真实意图，还是「画面外没法标」只能贴边框闭合")

    for i, deg in enumerate(degs):
        if deg < 8:
            notes.append(
                f"V{i + 1} 偏转 {deg:.1f}°（近乎共线，疑似冗余顶点；导入器与运行期"
                f"都**不做多边形简化**，形状按原样使用）")

    if ratio > 0.45:
        notes.append(f"区域覆盖画面 {ratio * 100:.0f}%，范围偏大 —— 入侵/违规告警量会明显高于窄区域")
    return notes


# ------------------------------------------------------------------ 绘制
def _clamp(v: float, max_index: int) -> int:
    """归一化坐标 → 像素索引，并夹到合法范围（x=1.0 会落在帧右边界之外）。"""
    return int(min(max(0.0, v), max_index))


def draw_zones(frame: np.ndarray, items) -> np.ndarray:
    """items: [(rule_display, zone, color)]；归一化坐标按帧实际宽高缩放。

    填充用整幅 addWeighted 即可：overlay 与原图只有多边形内部不同，混合后
    外部像素数值不变。（cv2.addWeighted 没有 mask 参数，别按 imgproc 习惯传。）
    """
    out = frame.copy()
    h, w = out.shape[:2]
    for label, zone, color in items:
        # 映射与运行期 ZoneMatcher 完全一致：归一化 × 宽/高（不是 w-1），
        # 否则预览与判定之间会差 0.05%，在贴边位置足够让人误判「到底进没进区域」。
        pts = np.array([[_clamp(x * w, w - 1), _clamp(y * h, h - 1)]
                        for x, y in zone.points], dtype=np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, 0.28, out, 0.72, 0)
        cv2.polylines(out, [pts], True, color, 2, cv2.LINE_AA)
        # 标签画在**质心**（区域内部）而不是首顶点外侧：文字本身是着色像素，
        # 画在区域外会让人误读成「区域延伸到那里」——预览图不该制造这种歧义。
        text = f"{label}/{zone.id}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        M = cv2.moments(pts.astype(np.float32).reshape(-1, 1, 2))
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        bx1, by1 = _clamp(cx - tw // 2, w - 1), _clamp(cy - th // 2, h - 1)
        bx2, by2 = _clamp(cx + tw // 2, w - 1), _clamp(cy + th // 2 + 2, h - 1)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.putText(out, text, (bx1, by2 - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    return out


# ------------------------------------------------------------------ 数据来源
def zones_from_config(cfg: dict, camera_id: str) -> tuple[dict, list, list, list]:
    """按摄像头 active_rules 取规则：带区域的画图，没区域的报「全图」。"""
    cam = next((c for c in cfg.get("cameras") or []
                if str(c.get("id")) == str(camera_id)), None)
    if cam is None:
        raise SystemExit(f"摄像头不存在: {camera_id}"
                         f"（配置里有: {', '.join(str(c.get('id')) for c in cfg.get('cameras') or [])}）")
    rules = cfg.get("rules") or {}
    active = cam.get("active_rules") or [
        n for n in rules if not str(n).startswith(GENERATED_PREFIX)]

    zone_items, fullframe, missing = [], [], []
    for name in active:
        rule = rules.get(name)
        if rule is None:
            missing.append(name)
            continue
        zones = parse_zones(rule.get("zones"), where=f"rules.{name}")
        if zones:
            zone_items.append((rule.get("alert_type") or name, zones))
        else:
            fullframe.append(name)
    return cam, zone_items, fullframe, missing


def zones_from_area_file(path: Path, rule_map) -> list[tuple[str, list]]:
    """直接读平台标定文件（导入前预演）。"""
    from tools.area_import import DEFAULT_RULE_MAP, build_zones
    mapping = dict(DEFAULT_RULE_MAP)
    mapping.update(rule_map)
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit(f"{path}: 顶层必须是区域列表")
    return [(rule, parse_zones(zones, where=f"area.{rule}"))
            for rule, zones in build_zones(entries, mapping).items()]


# ------------------------------------------------------------------ 入口
def main() -> int:
    ap = argparse.ArgumentParser(description="区域轮廓预览 / 几何核对（不进检测管线）")
    ap.add_argument("--config", help="config.yaml（按摄像头 active_rules 取区域）")
    ap.add_argument("--camera", help="摄像头 id，与 --config 搭配")
    ap.add_argument("--area", type=Path, help="平台标定文件（如 2628-area.txt），无需 config")
    ap.add_argument("--rule-map", action="append", default=[], metavar="MINOR=RULE",
                    help="与 --area 搭配：minorId→规则名（默认 9=intrusion）")
    ap.add_argument("--image", type=Path, help="用现成快照画图（完全不碰流）")
    ap.add_argument("--out", type=Path, help="输出路径（默认 preview-<camera>.jpg）")
    ap.add_argument("--frames", type=int, default=12, help="跳过的开头帧数")
    ap.add_argument("--timeout", type=float, default=25.0, help="取帧超时秒数")
    ap.add_argument("--stats-only", action="store_true", help="只打印几何事实，不取帧不画图")
    args = ap.parse_args()

    if not args.config and not args.area:
        ap.error("--config 与 --area 至少给一个")
    if args.config and not args.camera:
        ap.error("--config 需要同时给 --camera")

    cam: dict = {}
    collected: list = []
    source_url = None
    if args.area:
        rm = {}
        for item in args.rule_map:
            k, _, v = item.partition("=")
            rm[int(k)] = v.strip()
        collected = zones_from_area_file(args.area, rm)
        source_url = None
    else:
        cfg = load_config(args.config)
        cam, collected, fullframe, missing = zones_from_config(cfg, args.camera)
        source_url = cam.get("rtsp_url")
        if fullframe:
            print(f"  该路全图判定（未绑区域）: {', '.join(fullframe)}")
        if missing:
            print(f"  ⚠ active_rules 引用了不存在的规则: {', '.join(missing)}")

    if not collected:
        print("该路没有任何带区域的规则 —— 无需核对轮廓。")
        print("若要给这路配区域: python3 tools/area_import.py "
              f"--area <平台文件> --camera {args.camera} --config {args.config}")
        return 1

    # 先取帧：几何里的「距边框多少像素」要按**实际帧高**算，按 1080 估会误导
    frame = None
    if not args.stats_only:
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise SystemExit(f"快照无法读取: {args.image}")
            print(f"使用快照: {args.image}")
        else:
            if not source_url:
                raise SystemExit("配置里该摄像头没有 rtsp_url，请用 --image 传快照")
            print(f"拉流取帧: {mask_rtsp_url(source_url)}  "
                  f"（给该 URI 增加一路并发订阅，只读 1 帧即退出）")
            frame = grab_frame(source_url, args.timeout, args.frames)
            if frame is None:
                print("  ✗ 取帧失败。可改用平台给的快照: --image snapshot.jpg",
                      file=sys.stderr)

    ref_h = frame.shape[0] if frame is not None else 1080
    ref_w = frame.shape[1] if frame is not None else 1920
    print("区域几何:")
    warnings: list[str] = []
    for rule, zones in collected:
        print(f"  [{rule}]")
        for zone in zones:
            warnings.extend(describe_zone(zone, ref_height=ref_h, ref_width=ref_w))
    for w in warnings:
        print(f"  ⚠ {w}")

    if args.stats_only:
        return 0
    if frame is None:
        return 2

    h, w = frame.shape[:2]
    print(f"画面尺寸: {w}x{h}  区域坐标按此尺寸线性缩放（与 mux 1920x1080 的相对位置一致）")
    ratio = w / h
    if abs(ratio - 16 / 9) > 0.02:
        print(f"  注: 宽高比 {ratio:.3f} ≠ 16:9，检测管线会把它拉伸到 1920x1080；"
              f"归一化坐标对线性拉伸不变，本图仍对齐")

    items = []
    for i, (rule, zones) in enumerate(collected):
        color = PALETTE[i % len(PALETTE)]
        for zone in zones:
            items.append((rule, zone, color))

    out = draw_zones(frame, items)
    dest = args.out or Path(f"preview-{args.camera or 'zone'}.jpg")
    if not cv2.imwrite(str(dest), out):
        raise SystemExit(f"写出失败: {dest}")
    print(f"\n✓ 已输出 {dest} —— 请人眼确认区域是否贴合实际作业面/禁区边界，"
          f"再进影子实例测告警节奏")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
