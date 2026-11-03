#!/usr/bin/env python3
"""指定摄像头短时拉流+检人，只保留"有人"的帧（画框+conf 标注）存图。旁路工具，不碰生产。

在 helmet-dev 容器 /tmp/sandbox 内运行（GPU2）：
    python3 collect_cam.py --config config.yaml --model /tmp/p.pt --id 1168 --out /tmp/hits
"""
import argparse
import os
import time

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "ERROR")
import cv2
import yaml
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--model", default="/tmp/p.pt")
ap.add_argument("--id", required=True, help="摄像头ID（按 config.yaml cameras.id 匹配）")
ap.add_argument("--conf", type=float, default=0.25, help="算“有人”的检出下限（引擎门口径）")
ap.add_argument("--rounds", type=int, default=3, help="最多重试拉流轮数（每轮限时 30s）")
ap.add_argument("--stride", type=int, default=12, help="每 N 帧检测一次（≈1s）")
ap.add_argument("--max-keep", type=int, default=10, help="最多保留几张有人帧")
ap.add_argument("--out", default="hits")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cams = yaml.safe_load(open(a.config, encoding="utf-8"))["cameras"]
cam = next(c for c in cams if str(c["id"]).strip() == str(a.id).strip())
url = cam["rtsp_url"]
print(f"cam={a.id} ({cam.get('name','')}) {url}")

model = YOLO(a.model)
kept = 0
for rd in range(1, a.rounds + 1):
    if kept >= a.max_keep:
        break
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[round {rd}] 打开失败，2s 后重试", flush=True)
        time.sleep(2)
        continue
    n, t0 = 0, time.monotonic()
    while time.monotonic() - t0 < 30 and kept < a.max_keep:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % a.stride:
            continue
        det = model(frame, conf=a.conf, classes=[0], verbose=False)[0]
        if len(det.boxes) == 0:
            continue
        kept += 1
        vis = frame.copy()
        for b in det.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            c = float(b.conf)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 210, 0), 2)
            cv2.putText(vis, f"{c:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 210, 0), 2)
        f = os.path.join(a.out, f"cam{a.id}_{time.strftime('%H%M%S')}_k{kept:02d}.jpg")
        cv2.imwrite(f, vis)
        print(f"save {f} boxes={len(det.boxes)}", flush=True)
    cap.release()
    print(f"[round {rd}] 读{n}帧 累计保留{kept}", flush=True)

print(f"done kept={kept}")
