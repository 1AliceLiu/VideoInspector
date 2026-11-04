#!/usr/bin/env python3
"""实时吸烟判断记录（旁路，不触发告警、不改生产输出）。

对指定摄像头的实时 RTSP 画面，每隔 --every 秒：
  1. yolo26n 检出 person（conf ≥ 0.25，与生产引擎门一致）
  2. 对每个 person（最多 --max-persons 个）：
     - 小模型：yolo26m_smoking（线上同权重）在 person 整框上取 cigarette 最高 conf
     - 大模型：Qwen2.5-VL 单人裁图（生产 PROMPT，letterbox 512）取 P(yes)
  3. 不管是否触发告警，每个 person 一行写入 CSV（追加、即时 flush）

与生产的差异（写在表头备注里）：
  - 大模型这里是单帧裁图，生产是 4 帧连拍宫格；
  - 小模型取的是实时帧上的 person 整框，生产取的是触发帧属性值，二者口径接近但不完全相同。

在 helmet-dev 容器内运行（GPU2）：
  python3 live_smoke_judge.py --cam 2639 --url rtsp://118.140.234.166:8554/dahua1003143 --out /tmp/live-smoking
"""
import argparse
import base64
import csv
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "ERROR")
import cv2
import numpy as np
from ultralytics import YOLO

VLLM = "http://vllm:8013/v1/chat/completions"
VLM_MODEL = "/model/Qwen2.5-VL-7B-Instruct-AWQ"
# 与 server/alert/vlm_review.py 的 PROMPT 完全一致（单人裁图时作为单帧使用）
PROMPT = ("These 4 images are 4 consecutive cropped frames of the same person "
          "(top-left, top-right, bottom-left, bottom-right). "
          "Is this person smoking (holding a cigarette or having one in their mouth, "
          "whether lit or not; holding it between the fingers also counts)? "
          "Answer with exactly one word: yes or no.")
HEAD = ["北京时间", "摄像头ID", "采样序号", "人序号", "人框conf", "人框bbox",
        "小模型烟头conf", "大模型判定", "大模型P(yes)", "平均分(生产公式)", "裁图"]
NOTE = ("# 实时吸烟判断记录（旁路）\n\n"
        "> 每个 person 一行；小模型=烟头 conf（整框）；大模型=单帧裁图 P(yes)；"
        "平均分=(小模型+大模型)/2 仅供对照。生产大模型为 4 帧宫格，口径略有差异。\n\n")


def bj_now():
    return datetime.now(timezone.utc) + timedelta(hours=8)


def letterbox(img, size, pad=0.0, bbox=None):
    """与生产一致的裁剪：bbox 外扩 pad 后等比缩放，灰边补齐到 size×size。"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad)); x2 = min(w, int(x2 + bw * pad))
    y1 = max(0, int(y1 - bh * pad)); y2 = min(h, int(y2 + bh * pad))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    s = min(size / cw, size / ch)
    rz = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))),
                    interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, np.uint8)
    oy, ox = (size - rz.shape[0]) // 2, (size - rz.shape[1]) // 2
    canvas[oy:oy + rz.shape[0], ox:ox + rz.shape[1]] = rz
    return canvas


def ask_vlm(jpeg, timeout=8.0):
    body = {"model": VLM_MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                            + base64.b64encode(jpeg).decode()}}]}],
        "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20}
    req = urllib.request.Request(VLLM, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    c = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"][0]
    word = str(c["message"]["content"]).strip().lower()
    verdict = "yes" if word.startswith("y") else "no"
    try:
        cands = c["logprobs"]["content"][0]["top_logprobs"]
        py = sum(math.exp(t["logprob"]) for t in cands
                 if str(t["token"]).strip().lower().startswith("y"))
        pn = sum(math.exp(t["logprob"]) for t in cands
                 if str(t["token"]).strip().lower().startswith("n"))
        p = py / (py + pn) if (py + pn) > 0 else (1.0 if verdict == "yes" else 0.0)
    except (KeyError, IndexError, TypeError):
        p = 1.0 if verdict == "yes" else 0.0
    return verdict, p


def open_cap(url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    # 读超时 20s：H.265 长 GOP 的首个 IDR 可能要等十几秒（实测 1924 首帧 >10s）
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 20000])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap if cap.isOpened() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True, help="摄像头ID（写入 CSV）")
    ap.add_argument("--url", required=True, help="RTSP 地址（与 deploy/config.yaml 一致）")
    ap.add_argument("--person-model", default="/tmp/p.pt")
    ap.add_argument("--smoke-model", default="/tmp/smoke.pt")
    ap.add_argument("--every", type=float, default=2.0, help="采样间隔（秒）")
    ap.add_argument("--max-persons", type=int, default=3, help="每次采样最多判定几个人")
    ap.add_argument("--duration", type=int, default=7200, help="运行时长（秒）")
    ap.add_argument("--out", default="/tmp/live-smoking")
    a = ap.parse_args()

    crop_dir = os.path.join(a.out, "crops")
    os.makedirs(crop_dir, exist_ok=True)
    csv_path = os.path.join(a.out, f"live-smoking-{a.cam}.csv")
    md_path = os.path.join(a.out, f"live-smoking-{a.cam}-latest.md")
    new_file = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="", encoding="utf-8-sig")
    wr = csv.writer(fcsv)
    if new_file:
        wr.writerow(HEAD)
        fcsv.flush()

    person_m = YOLO(a.person_model)
    smoke_m = YOLO(a.smoke_model)
    names = smoke_m.names if isinstance(smoke_m.names, dict) else dict(enumerate(smoke_m.names))
    cig_ids = [i for i, n in names.items() if "cig" in str(n).lower() or "smok" in str(n).lower()]

    recent = []          # 最近行，用于 latest.md
    samples = persons_n = 0
    t_end = time.monotonic() + a.duration
    cap = open_cap(a.url)
    last_sample = 0.0
    print(f"live judge cam={a.cam} every={a.every}s max_persons={a.max_persons} "
          f"duration={a.duration}s csv={csv_path}", flush=True)

    while time.monotonic() < t_end:
        if cap is None:
            print(f"[{a.cam}] 打开失败，3s 后重试", flush=True)
            time.sleep(3)
            cap = open_cap(a.url)
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{a.cam}] 读帧失败，重连", flush=True)
            cap.release(); cap = None
            continue
        now = time.monotonic()
        if now - last_sample < a.every:
            continue
        last_sample = now
        samples += 1
        ts = bj_now().strftime("%Y-%m-%d %H:%M:%S")

        det = person_m(frame, conf=0.25, classes=[0], verbose=False)[0]
        boxes = sorted(((float(b.conf), tuple(map(int, b.xyxy[0]))) for b in det.boxes),
                       key=lambda t: -t[0])[:a.max_persons]
        print(f"[{a.cam}] 心跳 采样{samples} 帧{ts} 人数{len(boxes)}", flush=True)
        for k, (pconf, bbox) in enumerate(boxes):
            persons_n += 1
            x1, y1, x2, y2 = bbox
            # 小模型：整框裁剪（同 sgie process-mode=2），取 cigarette 最高 conf
            small_crop = letterbox(frame, 640, pad=0.0, bbox=bbox)
            small = 0.0
            if small_crop is not None:
                sd = smoke_m(small_crop, conf=0.01, imgsz=640, verbose=False)[0]
                small = max((float(b.conf) for b in sd.boxes if int(b.cls) in cig_ids),
                            default=0.0)
            # 大模型：生产同口径裁图（pad 0.15，512 letterbox）
            vlm_crop = letterbox(frame, 512, pad=0.15, bbox=bbox)
            verdict, p = "skipped", None
            if vlm_crop is not None:
                try:
                    ok_j, jpg = cv2.imencode(".jpg", vlm_crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    verdict, p = ask_vlm(jpg.tobytes())
                except Exception as e:           # 大模型不可用时只记小模型
                    print(f"[{a.cam}] vlm 失败: {e}", flush=True)
            avg = None if p is None else (small + p) / 2.0
            crop_name = ""
            if small >= 0.5 or (p is not None and p >= 0.5):
                crop_name = f"{a.cam}_{bj_now().strftime('%H%M%S')}_s{samples:05d}_p{k}.jpg"
                cv2.imwrite(os.path.join(crop_dir, crop_name), vlm_crop if vlm_crop is not None else frame)
            row = [ts, a.cam, samples, k, f"{pconf:.2f}", f"{x1},{y1},{x2},{y2}",
                   f"{small:.3f}", verdict, "-" if p is None else f"{p:.3f}",
                   "-" if avg is None else f"{avg:.3f}", crop_name]
            wr.writerow(row); fcsv.flush()
            recent.append(row)
            print(" ".join(str(x) for x in row[:10]), flush=True)

        recent = recent[-30:]
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(NOTE + f"最近 {len(recent)} 行（采样 {samples} 次，判定 {persons_n} 人次）\n\n")
            f.write("| " + " | ".join(HEAD) + " |\n|" + "|".join([":---"] * len(HEAD)) + "|\n")
            for r in recent:
                f.write("| " + " | ".join(str(x) for x in r) + " |\n")

    if cap is not None:
        cap.release()
    fcsv.close()
    print(f"done cam={a.cam} samples={samples} persons={persons_n}", flush=True)


if __name__ == "__main__":
    main()
