#!/usr/bin/env python3
"""VLM 复核旁路压测：生产零改动，逐路采全部摄像头的人框 → 线上同口径问大模型。

流程：解析 config.yaml 全部 rtsp_url → **逐路**短时拉流抽帧（避免 24 路并发软解
HEVC 压垮解码器）→ 原生 yolo26n @conf=0.15 检人 → 按线上同一套 vlm_review.REVIEWS
口径问 vLLM（安全帽/反光衣单帧即问；吸烟 IoU 跟踪攒 4 帧拼宫格），线程池并发
保持 vLLM 侧真实压力。CSV 每行 = 一次判定：conf 分桶 × 规则 × ✓确认/✗排除。

在 helmet-dev 容器内运行（需 torch/ultralytics/simplejpeg；已加入 safety-net 网络）：
    docker cp server/alert/vlm_review.py vlm_sidecar.py deploy/config.yaml helmet-dev:/tmp/
    docker exec -w /tmp helmet-dev python3 vlm_sidecar.py
    docker cp helmet-dev:/tmp/vlm_sidecar_result.csv ./vlm_sidecar_result.csv
"""

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "ERROR")   # 屏蔽解码器刷屏
import cv2
from ultralytics import YOLO

try:
    from server.alert import vlm_review          # 仓库内运行
except ImportError:
    import vlm_review                             # helmet-dev /tmp 平铺运行

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml", help="从其中解析全部 rtsp_url")
ap.add_argument("--model", default="/tmp/p.pt", help="原生 person 检测权重")
ap.add_argument("--url", default="http://vllm:8013/v1/chat/completions")
ap.add_argument("--vlm-model", default="/model/Qwen2.5-VL-7B-Instruct-AWQ")
ap.add_argument("--frames", type=int, default=40, help="每路读取帧数(含处理+跳过)")
ap.add_argument("--stride", type=int, default=6, help="每 N 帧检测一次(吸烟节奏≈240ms)")
ap.add_argument("--conf", type=float, default=0.15, help="检测下限")
ap.add_argument("--max-persons", type=int, default=3, help="每帧最多人数(按conf取高)")
ap.add_argument("--workers", type=int, default=8, help="vLLM 并发线程数")
ap.add_argument("--max-queries", type=int, default=600, help="总判定预算")
ap.add_argument("--out", default="vlm_sidecar_result.csv")
ap.add_argument("--temp", default="temp", help="可视化图输出目录（全帧+实际喂给大模型的图）")
a = ap.parse_args()
os.makedirs(a.temp, exist_ok=True)

BUCKET_COLORS = {"≥0.6(现门内)": (0, 200, 0), "0.3~0.6(降门新流入)": (0, 200, 200),
                 "0.15~0.3": (0, 0, 230)}    # BGR：绿/黄/红按 conf 桶给框上色

urls = re.findall(r"(rtsp://[^\s\"']+)", open(a.config, encoding="utf-8").read())
_s = set()
urls = [u for u in urls if not (u in _s or _s.add(u))]


def bucket(c):  # conf 分桶：对照线上 0.6 业务门
    return "≥0.6(现门内)" if c >= 0.6 else ("0.3~0.6(降门新流入)" if c >= 0.3 else "0.15~0.3")


def iou(ab, bb):
    iw = max(0, min(ab[2], bb[2]) - max(ab[0], bb[0]))
    ih = max(0, min(ab[3], bb[3]) - max(ab[1], bb[1]))
    inter = iw * ih
    area = (ab[2]-ab[0])*(ab[3]-ab[1]) + (bb[2]-bb[0])*(bb[3]-bb[1])
    return inter / max(1, area - inter)


model = YOLO(a.model)
pool = ThreadPoolExecutor(max_workers=a.workers)
csvf = open(a.out, "w", newline="", encoding="utf-8-sig")     # Excel 直开
writer = csv.writer(csvf)
writer.writerow(["北京时间", "规则", "conf", "conf分桶", "verdict", "结果", "帧数", "infer_ms", "图"])
stats, pending, budget = Counter(), [], [a.max_queries]


def drain(block=False):
    for f in [f for f in pending if block or f.done()]:
        if not block:
            pending.remove(f)
        else:
            pending.remove(f)
        row, key = f.result()
        writer.writerow(row)
        stats[key] += 1
    csvf.flush()


def ask_bucket(crops, prompt, expect, rule, conf, frames_n, tag):
    if budget[0] <= 0 or len(pending) >= a.workers * 4:   # 背压：防无限堆队列
        return
    budget[0] -= 1
    jpeg = vlm_review.build_grid(crops)
    img = f"{tag}_{rule}.jpg"
    open(os.path.join(a.temp, img), "wb").write(jpeg)     # 大模型实际看到的原图
    ts = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%m-%d %H:%M:%S")

    def job():
        t0 = time.monotonic()
        v = vlm_review.ask(a.url, a.vlm_model, jpeg, prompt, timeout=15)
        dt = round((time.monotonic() - t0) * 1000)
        ok = v is not None and v == expect
        row = (ts, rule, f"{conf:.2f}", bucket(conf), v or "skipped",
               "✓确认" if ok else "✗排除", frames_n, dt, img)
        return row, (rule, bucket(conf), ok)
    pending.append(pool.submit(job))


def submit_ppe(frame, bbox, rule, conf, tag):
    """v3 双人问提交：judge_ppe 在线程内跑（全图红框判人+部位特写判穿戴）。"""
    if budget[0] <= 0 or len(pending) >= a.workers * 4:
        return
    budget[0] -= 2
    ts = (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%m-%d %H:%M:%S")

    def job():
        person, wearing, violated, ms = vlm_review.judge_ppe(
            a.url, a.vlm_model, frame, bbox, rule, timeout=15)
        v = "yes" if violated else ("no" if person is not None else "skipped")
        row = (ts, rule, f"{conf:.2f}", bucket(conf), v,
               "✓确认" if v == "yes" else "✗排除", 1, ms, f"{tag}_{rule}.jpg")
        return row, (rule, bucket(conf), v == "yes")
    pending.append(pool.submit(job))


opened = 0
for idx, u in enumerate(urls):
    if budget[0] <= 0:
        break
    cap = cv2.VideoCapture(u, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # 尽量读最新帧，减积压
    if not cap.isOpened():
        print(f"[{idx}] 拉流失败: {u}", file=sys.stderr)
        continue
    opened += 1
    tracks, n = [], 0
    t0 = time.monotonic()
    while n < a.frames and budget[0] > 0:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % a.stride:
            continue
        det = model(frame, conf=a.conf, classes=[0], verbose=False)[0]
        tag0 = f"s{idx:02d}f{n:03d}"
        vis = frame.copy()                     # 人工核对用：全帧 + 所有检出框按 conf 桶上色
        for b in det.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0]); c = float(b.conf)
            col = BUCKET_COLORS[bucket(c)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"{c:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imwrite(os.path.join(a.temp, f"{tag0}_full.jpg"), vis)
        ppl = sorted(((float(b.conf), tuple(map(int, b.xyxy[0])))
                      for b in det.boxes), reverse=True)[:a.max_persons]
        for k, (conf, bbox) in enumerate(ppl):
            crop = vlm_review.crop_person(frame, bbox)
            if crop is None:
                continue
            tag = f"{tag0}p{k}"
            for rule in ("no_helmet", "no_vest"):          # v3：全图判人+部件判穿戴
                submit_ppe(frame, bbox, rule, conf, tag)
            tr = next((t for t in tracks if t["bbox"] and iou(t["bbox"], bbox) > 0.25), None)
            if tr is None:                                   # 吸烟：新起连拍轨
                tracks.append({"bbox": bbox, "crops": [crop], "miss": 0, "conf": conf, "tag": tag})
            elif len(tr["crops"]) < 4:
                tr["crops"].append(crop); tr["bbox"] = bbox; tr["miss"] = 0
                if len(tr["crops"]) == 4:                    # 攒满 → 宫格问
                    prompt, expect, _ = vlm_review.REVIEWS["smoking"]
                    ask_bucket(tr["crops"], prompt, expect, "smoking", tr["conf"], 4, tr["tag"])
                    tracks.remove(tr)
        for t in tracks[:]:                                  # 跟丢2轮：有几帧问几帧
            t["miss"] += 1
            if t["miss"] >= 2:
                prompt, expect, _ = vlm_review.REVIEWS["smoking"]
                ask_bucket(t["crops"], prompt, expect, "smoking", t["conf"], len(t["crops"]), t["tag"])
                tracks.remove(t)
        drain()
    for t in tracks:                                         # 换流前清残余轨道
        prompt, expect, _ = vlm_review.REVIEWS["smoking"]
        ask_bucket(t["crops"], prompt, expect, "smoking", t["conf"], len(t["crops"]), t["tag"])
    cap.release()
    print(f"[{idx}/{len(urls)}] {u.split('/')[-1]} 读{n}帧 "
          f"累计判定{sum(stats.values())} 耗时{time.monotonic()-t0:.0f}s", flush=True)

while pending:
    drain(block=True)

done = sum(stats.values())
print(f"\n===== 旁路压测汇总（在线{opened}/{len(urls)}路，{done} 次判定）=====")
for (rule, bkt, ok), cnt in sorted(stats.items()):
    print(f"{rule:10s} {bkt:22s} {'✓确认' if ok else '✗排除'}  {cnt}")
print(f"CSV: {a.out}")
