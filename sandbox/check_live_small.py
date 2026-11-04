#!/usr/bin/env python3
"""检查：live_smoke_judge 的小模型路径对“已知有烟”的图能不能出分。

用 1168 13:21:37 的去标注图（嘴里有烟，已人工确认），人框取红框反推值，
走与 live_smoke_judge.py 完全相同的 letterbox → yolo26m_smoking → cigarette 过滤。
在 helmet-dev 容器内运行：python3 check_live_small.py
"""
import importlib.util
import sys

import cv2
from ultralytics import YOLO

spec = importlib.util.spec_from_file_location("live", "/tmp/sandbox/live_smoke_judge.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)

m = YOLO("/tmp/smoke.pt")
names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
cig_ids = [i for i, n in names.items() if "cig" in str(n).lower() or "smok" in str(n).lower()]
print("names:", names, "cig_ids:", cig_ids, flush=True)

frame = cv2.imread("/tmp/known_smoke.jpg")
print("frame:", None if frame is None else frame.shape, flush=True)
bbox = (645, 792, 862, 1080)
crop = live.letterbox(frame, 640, pad=0.0, bbox=bbox)
print("crop:", None if crop is None else crop.shape, flush=True)
res = m(crop, conf=0.01, imgsz=640, verbose=False)[0]
print("raw boxes (cls, conf):", [(int(b.cls), round(float(b.conf), 3)) for b in res.boxes], flush=True)
small = max((float(b.conf) for b in res.boxes if int(b.cls) in cig_ids), default=0.0)
print("live 路径算出的小模型 conf =", round(small, 3), flush=True)
sys.exit(0)
