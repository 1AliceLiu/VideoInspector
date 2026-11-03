#!/usr/bin/env python3
"""对 smoke-recheck 的 4 张告警原图，用线上同权重 smoke 小模型重算吸烟置信度。

口径尽量贴生产：person 整框裁剪（不带 pad，DeepStream sgie process-mode=2 即整框）
→ 等比 letterbox 到 640 → yolo26m_smoking。输入用去掉 OSD 红框的 erased 图。

在 helmet-dev 容器内跑（GPU2）：
    python3 smoke_small_scores.py --dir /tmp/smoke-recheck --model /tmp/smoke.pt
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def locate_box(img):
    m = cv2.inRange(img, np.array((0, 0, 240)), np.array((40, 40, 255)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    area, (x, y, w, h, _) = max(((int(s[4]), tuple(map(int, s))) for s in stats[1:]),
                                key=lambda t: t[0])
    if w < 40 or h < 60 or area < 500:
        return None
    return max(0, x), max(0, y), x + w, y + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/smoke-recheck")
    ap.add_argument("--model", default="/tmp/smoke.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    model = YOLO(a.model)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    print("模型类别:", names)

    print("| 图 | person 框 | 吸烟类别 | 小模型 conf | 其他类别最高 conf |")
    print("| :--- | :--- | :---: | ---: | ---: |")
    for before in sorted(glob.glob(os.path.join(a.dir, "*_before.jpg"))):
        erased_path = before.replace("_before.jpg", "_erased.jpg")
        bbox = locate_box(cv2.imread(before))
        img = cv2.imread(erased_path)
        x1, y1, x2, y2 = bbox
        crop = img[max(0, y1):y2, max(0, x1):x2]
        det = model(crop, conf=0.01, imgsz=a.imgsz, verbose=False)[0]
        rows = sorted(((float(b.conf), names[int(b.cls)]) for b in det.boxes), reverse=True)
        cigt = next(((c, n) for c, n in rows if "cig" in n.lower() or "smok" in n.lower()), None)
        other = next(((c, n) for c, n in rows if not ("cig" in n.lower() or "smok" in n.lower())), None)
        name = os.path.basename(before).replace("_before.jpg", "")
        cig_s = f"{cigt[0]:.4f}" if cigt else "0（未检出）"
        cig_n = cigt[1] if cigt else "-"
        oth = f"{other[0]:.4f} ({other[1]})" if other else "-"
        print(f"| {name} | {bbox} | {cig_n} | {cig_s} | {oth} |")
        top3 = ", ".join(f"{n}:{c:.3f}" for c, n in rows[:3]) or "无任何检出"
        print(f"  top: {top3}")


if __name__ == "__main__":
    main()
