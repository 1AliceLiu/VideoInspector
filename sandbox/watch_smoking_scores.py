#!/usr/bin/env python3
"""盯指定摄像头的 smoking 告警：每条事件重算小模型 conf 与大模型 P(yes)，追加成表格。

跑在 helmet-dev 容器内（需 cv2/ultralytics；GPU2）：
    python3 watch_smoking_scores.py --cams 2639,1924 --duration 1800

输出（容器内 /tmp，宿主机由外层循环同步到 temp/）：
    smoking-confidence-<cams>.csv   # 一行一个事件
    smoking-confidence-<cams>.md    # Markdown 表格（每次追加后重写）
"""
import argparse
import base64
import csv
import json
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
from ultralytics import YOLO

PLATFORM = "http://192.168.1.24:6034/protal/ai-alarm"
VLLM = "http://vllm:8013/v1/chat/completions"
MODEL_VLM = "/model/Qwen2.5-VL-7B-Instruct-AWQ"
PROMPT = ("These 4 images are 4 consecutive cropped frames of the same person "
          "(top-left, top-right, bottom-left, bottom-right). "
          "Is this person smoking (holding a cigarette or having one in their mouth, "
          "whether lit or not; holding it between the fingers also counts)? "
          "Answer with exactly one word: yes or no.")
HEAD = ["时间(北京)", "摄像头ID", "摄像头名称", "告警ID", "小模型conf", "小模型来源",
        "大模型判定", "大模型P(yes)", "平均分(仅参考)", "门控>0.75", "证据图"]


def http_get(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def query(cam_id, start, end):
    qs = urllib.parse.urlencode({"pageno": 1, "pagesize": 50, "alarmType": "smoking",
                                 "cameraKeyword": cam_id,
                                 "startTime": start, "endTime": end})
    d = json.loads(http_get(f"{PLATFORM}/list?{qs}"))
    # cameraKeyword 是模糊匹配（1924 会命中名字相同的 1942），必须按 id 精确过滤
    return [a for a in d.get("res", {}).get("list", [])
            if str(a.get("cameraId")) == str(cam_id)]


def locate_box(img):
    m = cv2.inRange(img, (0, 0, 240), (40, 40, 255))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    area, (x, y, w, h, _) = max(((int(s[4]), tuple(map(int, s))) for s in stats[1:]),
                                key=lambda t: t[0])
    if w < 40 or h < 60 or area < 500:
        return None
    return x, y, x + w, y + h


def erase_osd(img, bbox, label="smoking"):
    h, w = img.shape[:2]
    if bbox is None:
        return img
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bx1, by1 = max(0, x1 - 16), max(0, y1 - 16)
    bx2, by2 = min(w, x2 + 16), min(h, y2 + 16)
    loose = cv2.inRange(img, np.array((0, 0, 180)), np.array((80, 80, 255)))
    mask = np.zeros((h, w), np.uint8)
    mask[by1:by2, bx1:bx2] = loose[by1:by2, bx1:bx2]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = y1 - 8 if y1 - 8 - th >= 0 else min(h - 1, y2 + th + 8)
    cv2.rectangle(mask, (max(0, x1 - 4), max(0, ty - th - 8)),
                  (min(w - 1, x1 + tw + 8), min(h - 1, ty + 8)), 255, -1)
    mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=1)
    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def crop_person(img, bbox, size=640, pad=0.05):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad)); x2 = min(w, int(x2 + bw * pad))
    y1 = max(0, int(y1 - bh * pad)); y2 = min(h, int(y2 + bh * pad))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return img
    ch, cw = crop.shape[:2]
    s = min(size / cw, size / ch)
    rz = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))),
                    interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, np.uint8)
    oy, ox = (size - rz.shape[0]) // 2, (size - rz.shape[1]) // 2
    canvas[oy:oy + rz.shape[0], ox:ox + rz.shape[1]] = rz
    return canvas


def vlm_score(jpeg_bytes, timeout=15):
    body = json.dumps({"model": MODEL_VLM, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                    + base64.b64encode(jpeg_bytes).decode()}}]}],
        "max_tokens": 1, "temperature": 0.0, "logprobs": True, "top_logprobs": 20}).encode()
    req = urllib.request.Request(VLLM, data=body,
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    c = r["choices"][0]
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


def process(alert, model, names, out_dir):
    raw = http_get(f"{PLATFORM}/evidence?path={urllib.parse.quote(alert['framePath'])}")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    bbox = locate_box(img)
    erased = erase_osd(img, bbox)
    crop = crop_person(erased, bbox) if bbox else erased

    det = model(crop, conf=0.01, imgsz=640, verbose=False)[0]
    cig = max((float(b.conf) for b in det.boxes
               if "cig" in str(names[int(b.cls)]).lower()
               or "smok" in str(names[int(b.cls)]).lower()), default=0.0)
    small_src = "重算(存档图)"
    if alert.get("topConfidence") is not None:      # 平台已存的小模型分数优先
        cig, small_src = float(alert["topConfidence"]), "平台存值"

    ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    verdict, p = vlm_score(jpg.tobytes())
    avg = (cig + p) / 2.0

    tag = f"c{alert['cameraId']}_{alert['eventTime'].replace(':', '').replace(' ', '_')}"
    cv2.imwrite(os.path.join(out_dir, f"{tag}_crop.jpg"), crop)
    return [alert["eventTime"], alert["cameraId"], alert.get("cameraName", ""),
            alert["id"], f"{cig:.3f}", small_src, verdict, f"{p:.3f}",
            f"{avg:.3f}", "推送" if avg > 0.75 else "丢弃", f"{tag}_crop.jpg"]


def write_outputs(path_csv, path_md, rows):
    with open(path_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(HEAD); w.writerows(rows)
    lines = ["# 2639/1924 smoking 事件置信度（小模型 + 大模型）", "",
             "> 小模型=烟头检测 conf（优先平台存值，否则用存档图重算）；"
             "大模型=Qwen2.5-VL 单帧 P(yes)；avg 仅参考，生产门控是 4 帧宫格。", "",
             "| " + " | ".join(HEAD) + " |",
             "|" + "|".join([":---"] * len(HEAD)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    if not rows:
        lines += ["", "_暂无 smoking 事件_"]
    open(path_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="2639,1924")
    ap.add_argument("--interval", type=int, default=20, help="轮询间隔秒")
    ap.add_argument("--duration", type=int, default=1800, help="运行时长秒")
    ap.add_argument("--out", default="/tmp")
    ap.add_argument("--model", default="/tmp/smoke.pt")
    a = ap.parse_args()
    cams = [c.strip() for c in a.cams.split(",") if c.strip()]
    os.makedirs(a.out, exist_ok=True)
    path_csv = os.path.join(a.out, f"smoking-confidence-{'-'.join(cams)}.csv")
    path_md = os.path.join(a.out, f"smoking-confidence-{'-'.join(cams)}.md")

    model = YOLO(a.model)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))

    rows, seen = [], set()
    if os.path.exists(path_csv):
        with open(path_csv, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows.append([r[h] for h in HEAD]); seen.add(str(r["告警ID"]))
    write_outputs(path_csv, path_md, rows)

    bj = datetime.now(timezone.utc) + timedelta(hours=8)
    day = bj.strftime("%Y-%m-%d")
    start = f"{day} 00:00:00"   # 当天全部（北京时间），靠 seen 去重
    end = f"{day} 23:59:59"
    t_end = time.monotonic() + a.duration
    print(f"watch cams={cams} every {a.interval}s, {a.duration}s; "
          f"已有 {len(rows)} 行", flush=True)
    while time.monotonic() < t_end:
        for cam in cams:
            try:
                alerts = query(cam, start, end)
            except Exception as e:
                print(f"poll {cam} failed: {e}", flush=True)
                continue
            for al in reversed(alerts):          # 旧→新
                if str(al["id"]) in seen:
                    continue
                try:
                    row = process(al, model, names, a.out)
                except Exception as e:
                    print(f"process {al['id']} failed: {e}", flush=True)
                    continue
                seen.add(str(al["id"]))
                rows.append(row)
                write_outputs(path_csv, path_md, rows)
                print("NEW", row[:5], flush=True)
        time.sleep(a.interval)
    print(f"done rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
