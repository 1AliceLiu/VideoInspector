#!/usr/bin/env python3
"""VLM 复核旁路实验：生产零改动，测「低 conf 人框在复核口径下会被判成什么」。

流程：拉摄像头原始 RTSP（无 OSD 污染）→ 原生 yolo26n @conf=0.15 检人 → 按线上
同一套 vlm_review.REVIEWS 口径问 vLLM：
  安全帽/反光衣 → 单帧即问；吸烟 → IoU 跟踪同人攒 4 帧连拍拼宫格再问。
CSV 每行 = 一次复核判定（conf 分桶 × 规则 × ✓确认/✗排除），用于回答：
  「如果把 person_conf_threshold 降到 0.3，大模型挡不挡得住涌进来的边界框？」

在 helmet-dev 容器内运行（需 torch/ultralytics/simplejpeg；需已加入 safety-net 网络）：
    docker cp server/alert/vlm_review.py vlm_sidecar.py helmet-dev:/tmp/
    docker exec -w /tmp helmet-dev python3 vlm_sidecar.py --rtsp rtsp://<摄像头源>
    docker cp helmet-dev:/tmp/vlm_sidecar_result.csv ./
"""

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from server.alert import vlm_review          # 仓库内运行
except ImportError:
    import vlm_review                             # helmet-dev /tmp 平铺运行

ap = argparse.ArgumentParser()
ap.add_argument("--rtsp", required=True)
ap.add_argument("--model", default="/tmp/p.pt", help="原生 person 检测权重")
ap.add_argument("--url", default="http://vllm:8013/v1/chat/completions")
ap.add_argument("--vlm-model", default="/model/Qwen2.5-VL-7B-Instruct-AWQ")
ap.add_argument("--seconds", type=float, default=120, help="取流时长")
ap.add_argument("--stride", type=int, default=4, help="每 N 帧处理一次(吸烟连拍节奏)")
ap.add_argument("--conf", type=float, default=0.15, help="检测下限")
ap.add_argument("--max-persons", type=int, default=4, help="每帧最多处理的人数(按conf取高)")
ap.add_argument("--max-queries", type=int, default=120, help="总判定预算(控时长)")
ap.add_argument("--out", default="vlm_sidecar_result.csv")
a = ap.parse_args()

def bucket(c):  # conf 分桶：对照线上 0.6 业务门
    return "≥0.6(现门内)" if c >= 0.6 else ("0.3~0.6(降门新流入)" if c >= 0.3 else "0.15~0.3")

def iou(AB, bb):
    ax1, ay1, ax2, ay2 = AB; bx1, by1, bx2, by2 = bb
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (max(1, (ax2-ax1)*(ay2-ay1)) + max(1, (bx2-bx1)*(by2-by1)) - inter)

model = YOLO(a.model)
csvf = open(a.out, "w", newline="", encoding="utf-8-sig")     # Excel 直开
writer = csv.writer(csvf)
writer.writerow(["北京时间", "规则", "conf", "conf分桶", "verdict", "结果", "帧数", "infer_ms"])
stats = Counter()
budget = [a.max_queries]

def ask_bucket(bbox, conf, crops, prompt, expect, rule, frames_n):
    if budget[0] <= 0:
        return
    budget[0] -= 1
    jpeg = vlm_review.build_grid(crops)
    t0 = time.monotonic()
    v = vlm_review.ask(a.url, a.vlm_model, jpeg, prompt, timeout=10)
    dt = round((time.monotonic() - t0) * 1000)
    ok = v is not None and v == expect
    ts = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%m-%d %H:%M:%S")
    writer.writerow([ts, rule, f"{conf:.2f}", bucket(conf), v or "skipped",
                     "✓确认" if ok else "✗排除", frames_n, dt])
    csvf.flush()
    stats[(rule, bucket(conf), ok)] += 1

cap = cv2.VideoCapture(a.rtsp)
if not cap.isOpened():
    sys.exit(f"拉流失败: {a.rtsp}")
tracks = []            # 吸烟连拍轨道: {bbox, crops[], miss}
end = time.monotonic() + a.seconds
frame_i = 0
while time.monotonic() < end and budget[0] > 0:
    ok, frame = cap.read()
    if not ok:
        break
    frame_i += 1
    if frame_i % a.stride:
        continue
    det = model(frame, conf=a.conf, classes=[0], verbose=False)[0]
    ppl = sorted(((float(b.conf), tuple(map(int, b.xyxy[0]))) for b in det.boxes), reverse=True)[:a.max_persons]
    for conf, bbox in ppl:
        crop = vlm_review.crop_person(frame, bbox)
        if crop is None:
            continue
        for rule in ("no_helmet", "no_vest"):          # 静态属性：单帧即问
            prompt, expect, _ = vlm_review.REVIEWS[rule]
            ask_bucket(bbox, conf, [crop], prompt, expect, rule, 1)
        tr = next((t for t in tracks if t["bbox"] and iou(t["bbox"], bbox) > 0.25), None)
        if tr is None:                                   # 吸烟：新起连拍轨
            tracks.append({"bbox": bbox, "crops": [crop], "miss": 0})
        elif len(tr["crops"]) < 4:
            tr["crops"].append(crop); tr["bbox"] = bbox; tr["miss"] = 0
            if len(tr["crops"]) == 4:                    # 攒满4帧 → 宫格问
                prompt, expect, _ = vlm_review.REVIEWS["smoking"]
                ask_bucket(bbox, conf, tr["crops"], prompt, expect, "smoking", 4)
                tracks.remove(tr)
    for t in tracks[:]:                                  # 跟丢2轮：有几帧问几帧
        t["miss"] += 1
        if t["miss"] >= 2:
            prompt, expect, _ = vlm_review.REVIEWS["smoking"]
            ask_bucket(t["bbox"], 0.5, t["crops"], prompt, expect, "smoking", len(t["crops"]))
            tracks.remove(t)
cap.release()

print(f"\n===== 旁路实验汇总（{sum(stats.values())} 次判定）=====")
for (rule, bkt, ok), n in sorted(stats.items()):
    print(f"{rule:10s} {bkt:22s} {'✓确认' if ok else '✗排除'}  {n}")
print(f"CSV: {a.out}")
