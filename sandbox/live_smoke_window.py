#!/usr/bin/env python3
"""时序窗口吸烟判定（沙盒旁路，不触发告警、不改生产输出）。

每个 person 以 0.5s 间隔持续取帧（同一条 IoU≥0.3 视为同一人）：
  - 小模型 yolo26m_smoking：每个采样帧在 person 整框（640 紧裁，同旧口径）打分，
    conf≥0.75（生产 rule attribute_threshold）记一票，≥2 票举手。
  - 大模型 Qwen2.5-VL：窗口满 5s（或人丢失且已≥4 帧）终审一次，
    取窗口内至多 8 帧时序裁图（512 pad0.15，同生产裁图口径）一次多图调用取 P(yes)。

可视化产物：temp/live-smoking-win/ 下的 CSV（每窗口一行）+ 5s 窗口视频
（默认只存有信号的窗口：P(yes)≥0.4 或有小模型票；要全存改 SAVE_MIN_P=-1）。

在 helmet-dev 容器内运行（与旧脚本相同的 /tmp 权重）：
  python3 live_smoke_window.py --cam 2639 --url rtsp://118.140.234.166:8554/dahua1003143 --out /tmp/live-smoking-win
"""
import argparse
import base64
import csv
import json
import math
import os
import time
import urllib.request
from datetime import datetime, timedelta, timezone

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "ERROR")
import cv2
import numpy as np
from ultralytics import YOLO

VLLM = "http://vllm:8013/v1/chat/completions"
VLM_MODEL = "/model/Qwen2.5-VL-7B-Instruct-AWQ"
PROMPT_WIN = ("These {n} images are {n} frames of the same person sampled at 0.5-second "
              "intervals over a 5-second window, in chronological order (first = earliest). "
              "Judge by the motion across frames: is this person smoking (hand repeatedly "
              "raising a cigarette to the mouth, or holding a cigarette in fingers / mouth, "
              "whether lit or not)? Answer with exactly one word: yes or no.")
# 单帧对照轨：与旧脚本/昨天 CSV 完全同口径（生产宫格 PROMPT 当单帧用），保证可比
PROMPT_1F = ("These 4 images are 4 consecutive cropped frames of the same person "
             "(top-left, top-right, bottom-left, bottom-right). "
             "Is this person smoking (holding a cigarette or having one in their mouth, "
             "whether lit or not; holding it between the fingers also counts)? "
             "Answer with exactly one word: yes or no.")
SMOKE_VOTE_TH = 0.75   # 生产 rules.smoking.attribute_threshold
VOTE_K = 2             # 举手门槛：每窗口 ≥2 票（N 见 CSV 采样数列）
SAVE_MIN_P = 0.4       # 视频保存门槛；改 -1 = 全存
HEAD = ["北京时间", "摄像头ID", "窗口序号", "跟踪ID", "人框conf", "人框bbox", "采样数",
        "小模型max", "小模型≥0.75票数", "举手结果", "大模型判定", "大模型P(yes)",
        "单帧P(yes)", "翻转", "平均分(生产公式)", "视频"]
NOTE = ("# 时序窗口吸烟判定（沙盒旁路，双轨：时序 8 帧 + 单帧对照）\n\n"
        "> 每 person 每 5s 一个窗口一行；小模型=每 0.5s 整框打分，≥0.75 一票，≥%d 票举手；"
        "大模型判定=窗口内≤8 帧时序终审 P(yes)；单帧P=同窗口中间帧生产原口径；翻转=两轨对比；"
        "平均分=(时序小max+时序大)/2 仅对照。视频存 P≥%.1f/有票/单帧≥0.5 的窗口。\n\n"
        % (VOTE_K, SAVE_MIN_P))


def bj_now():
    return datetime.now(timezone.utc) + timedelta(hours=8)


def letterbox(img, size, pad=0.0, bbox=None):
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


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / area if area > 0 else 0.0


def pick_frames(samples, k=8):
    """窗口内按时间均匀取至多 k 帧的下标。"""
    n = len(samples)
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def parse_pyes(c):
    """从 vllm 响应取 verdict + P(yes)（logprobs 还原，同旧口径）。"""
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


def ask_vlm_multi(jpegs, prompt=None, timeout=15.0):
    content = [{"type": "text",
                "text": prompt if prompt else PROMPT_WIN.format(n=len(jpegs))}]
    for j in jpegs:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,"
                                             + base64.b64encode(j).decode()}})
    body = {"model": VLM_MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20}
    req = urllib.request.Request(VLLM, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    c = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["choices"][0]
    return parse_pyes(c)


def open_cap(url):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                            cv2.CAP_PROP_READ_TIMEOUT_MSEC, 20000])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 终审阻塞时不积压旧帧（同旧脚本）
    return cap


def write_video(path, crops):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (512, 512))
    if not vw.isOpened():                      # mp4v 不可用则退 MJPG/avi
        path = path[:-4] + ".avi"
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 2.0, (512, 512))
    for c in crops:
        vw.write(c)
    vw.release()
    return os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--person-model", default="/tmp/p.pt")
    ap.add_argument("--smoke-model", default="/tmp/smoke.pt")
    ap.add_argument("--period", type=float, default=0.5, help="采样间隔（秒）")
    ap.add_argument("--window", type=float, default=5.0, help="终审窗口时长（秒）")
    ap.add_argument("--max-persons", type=int, default=3)
    ap.add_argument("--duration", type=int, default=7200)
    ap.add_argument("--out", default="/tmp/live-smoking-win")
    a = ap.parse_args()

    video_dir = os.path.join(a.out, "videos")
    os.makedirs(video_dir, exist_ok=True)
    csv_path = os.path.join(a.out, f"win-{a.cam}.csv")
    md_path = os.path.join(a.out, f"win-{a.cam}-latest.md")
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

    tracks = {}            # tid -> dict(last_seen, pconf, bbox, samples:[{t,small,sc,vc}])
    next_tid = [0]
    recent = []
    windows_n = 0
    t_end = time.monotonic() + a.duration
    cap = open_cap(a.url)
    last_sample = 0.0
    print(f"window judge cam={a.cam} period={a.period}s window={a.window}s "
          f"max_persons={a.max_persons} duration={a.duration}s csv={csv_path}", flush=True)

    def judge(tid, tr):
        """窗口终审：一次多图 VLM 调用 + CSV 行 + 按门槛存视频。"""
        nonlocal windows_n
        windows_n += 1
        samples = tr["samples"]
        votes = sum(1 for s in samples if s["small"] >= SMOKE_VOTE_TH)
        smax = max(s["small"] for s in samples)
        idx = pick_frames(samples)
        verdict, p = "skipped", None
        p1 = None
        try:
            jpegs = [cv2.imencode(".jpg", samples[i]["vc"],
                                  [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                     for i in idx]
            t0 = time.monotonic()
            verdict, p = ask_vlm_multi(jpegs)
            ms = int((time.monotonic()-t0)*1000)
            # 单帧对照轨：同窗口中间帧、生产原 PROMPT 当单帧（同昨天口径）
            _, p1 = ask_vlm_multi([jpegs[len(idx)//2]], prompt=PROMPT_1F)
            print(f"[{a.cam}] 终审 {len(jpegs)}图 {ms}ms 单帧对照 {int((time.monotonic()-t0)*1000)-ms}ms", flush=True)
        except Exception as e:
            print(f"[{a.cam}] vlm 失败: {e}", flush=True)
        avg = None if p is None else (smax + p) / 2.0
        vid = ""
        if p is not None and (p >= SAVE_MIN_P or votes > 0 or (p1 and p1 >= 0.5)):
            vid = write_video(os.path.join(
                video_dir, f"{a.cam}_w{windows_n:05d}_t{tid}.mp4"),
                [samples[i]["vc"] for i in range(len(samples))])
        x1, y1, x2, y2 = tr["bbox"]
        if p is None or p1 is None:
            flip = "-"
        elif p1 >= 0.5 and p < 0.5:
            flip = "时序压制"
        elif p1 < 0.5 and p >= 0.5:
            flip = "时序放行"
        else:
            flip = "一致"
        row = [bj_now().strftime("%Y-%m-%d %H:%M:%S"), a.cam, windows_n, tid,
               f"{tr['pconf']:.2f}", f"{x1},{y1},{x2},{y2}", len(samples),
               f"{smax:.3f}", votes, "yes" if votes >= VOTE_K else "no",
               verdict, "-" if p is None else f"{p:.3f}",
               "-" if p1 is None else f"{p1:.3f}", flip,
               "-" if avg is None else f"{avg:.3f}", vid]
        wr.writerow(row); fcsv.flush()
        recent.append(row)
        print(" ".join(str(x) for x in row[:15]), flush=True)
        tr["samples"] = []   # 人还在就开下一窗口，跟踪关系保留

    while time.monotonic() < t_end:
        if not cap.isOpened():
            time.sleep(3)
            cap = open_cap(a.url)
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release(); cap = open_cap(a.url)
            continue
        now = time.monotonic()
        if now - last_sample < a.period:
            continue
        last_sample = now
        ts = now

        det = person_m(frame, conf=0.25, classes=[0], verbose=False)[0]
        dets = sorted(((float(b.conf), tuple(map(int, b.xyxy[0]))) for b in det.boxes),
                      key=lambda t: -t[0])[:a.max_persons]

        taken = set()
        for pconf, bbox in dets:
            best_tid, best_iou = None, 0.3
            for tid, tr in tracks.items():
                if tid in taken:
                    continue
                v = iou(tr["bbox"], bbox)
                if v > best_iou:
                    best_tid, best_iou = tid, v
            if best_tid is None:
                best_tid = next_tid[0]; next_tid[0] += 1
                tracks[best_tid] = {"pconf": pconf, "bbox": bbox,
                                    "last_seen": ts, "samples": []}
            tr = tracks[best_tid]
            taken.add(best_tid)
            tr.update(pconf=pconf, bbox=bbox, last_seen=ts)
            x1, y1, x2, y2 = bbox
            small = 0.0
            sc = letterbox(frame, 640, pad=0.0, bbox=bbox)
            if sc is not None:
                sd = smoke_m(sc, conf=0.01, imgsz=640, verbose=False)[0]
                small = max((float(b.conf) for b in sd.boxes if int(b.cls) in cig_ids),
                            default=0.0)
            vc = letterbox(frame, 512, pad=0.15, bbox=bbox)
            if vc is None:
                continue
            tr["samples"].append({"t": ts, "small": small, "vc": vc})
            if ts - tr["samples"][0]["t"] >= a.window:
                judge(best_tid, tr)

        # 跟丢 >1.5s：若已有 ≥4 帧样本则提前终审，否则丢弃
        for tid in list(tracks):
            tr = tracks[tid]
            if ts - tr["last_seen"] > 1.5:
                if len(tr["samples"]) >= 4:
                    judge(tid, tr)
                del tracks[tid]

        recent = recent[-30:]
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(NOTE + f"最近 {len(recent)} 个窗口（已终审 {windows_n}）\n\n")
            f.write("| " + " | ".join(HEAD) + " |\n|" + "|".join([":---"] * len(HEAD)) + "|\n")
            for r in recent:
                f.write("| " + " | ".join(str(x) for x in r) + " |\n")

    cap.release()
    fcsv.close()
    print(f"done cam={a.cam} windows={windows_n}", flush=True)


if __name__ == "__main__":
    main()
