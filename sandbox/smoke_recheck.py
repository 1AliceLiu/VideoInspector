#!/usr/bin/env python3
"""对平台已落库的 smoking 告警图做"去掉 OSD 标注 → 大模型复核"。

流程：
  1. GET /protal/ai-alarm/list 按时间/类型查告警，取 detectionsJson 的 bbox
  2. GET /protal/ai-alarm/evidence 下载证据图（原始帧 + 红色 OSD 框 + smoking 标签）
  3. 按 _render_evidence 的已知几何擦除 OSD（红框线宽4 / 标签 HERSHEY_SIMPLEX 0.7,2），
     用 cv2.inpaint(TELEA) 修补，只动标注像素，不碰画面内容
  4. 按 bbox 外扩裁人 → letterbox → 交给 Qwen2.5-VL 单帧问吸烟
  5. before / erased / crop 三张图落盘，便于人工核对

用法（宿主机即可，依赖 cv2/numpy）：
    python3 sandbox/smoke_recheck.py --out temp/smoke-recheck
"""
import argparse
import base64
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime

import cv2
import numpy as np

PLATFORM = "http://192.168.1.24:6034/protal/ai-alarm"
VLLM = "http://vllm:8013/v1/chat/completions"   # 容器内；宿主机跑用 --vllm http://127.0.0.1:8013/...
MODEL = "/model/Qwen2.5-VL-7B-Instruct-AWQ"
PROMPT = ("下图为一次抽烟告警的目标裁剪图。若此人在吸烟（手持香烟/嘴中含烟，点燃与否、"
          "夹在指间均算），答 yes；否则（只是抬手、拿工具、喝水、非真人等）答 no。"
          "只回答一个词：yes 或 no。")


def http_get(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def query_alerts(start, end, alarm_type="smoking"):
    qs = urllib.parse.urlencode({"pageno": 1, "pagesize": 100, "alarmType": alarm_type,
                                 "startTime": start, "endTime": end})
    d = json.loads(http_get(f"{PLATFORM}/list?{qs}"))
    return d.get("res", {}).get("list", [])


def locate_box(img):
    """在无 detectionsJson 的旧告警图上，按纯红 OSD 找回人物框 bbox。

    生产红框 BGR=(0,0,255) 线宽 4；用紧阈值取最大连通域（框线面积远大于画面红色噪点）。
    返回 (x1,y1,x2,y2) 或 None。
    """
    m = cv2.inRange(img, np.array((0, 0, 240)), np.array((40, 40, 255)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    area, (x, y, w, h, _) = max(((int(s[4]), tuple(map(int, s))) for s in stats[1:]),
                                key=lambda t: t[0])
    if w < 40 or h < 60 or area < 500:      # 人框至少这么大；否则认为是噪声
        return None
    return x, y, x + w, y + h


def erase_osd(img, bbox, label="smoking"):
    """擦除指定 bbox 的红框线与标签，返回 (擦除图, 残留红像素数)。

    框线在 JPEG 里会晕开成 8px 左右的红色带，因此用颜色阈值取 bbox 周围
    16px 带内的红像素（含晕边）；标签按生产固定落位整块擦除。带外的红色
    水马等画面内容不动。
    """
    h, w = img.shape[:2]
    if bbox is None:
        return img, -1
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
    erased = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    left = cv2.inRange(erased[by1:by2, bx1:bx2], np.array((0, 0, 200)),
                       np.array((80, 80, 255)))
    return erased, int((left > 0).sum())


def crop_person(img, bbox, size=512, pad=0.15):
    """与 vlm_review.crop_person 同口径：外扩 + 等比 letterbox 灰边。"""
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
    resized = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, np.uint8)
    oy, ox = (size - resized.shape[0]) // 2, (size - resized.shape[1]) // 2
    canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
    return canvas


def ask_vlm(jpeg, timeout=30):
    body = {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                            + base64.b64encode(jpeg).decode()}}]}],
        "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(VLLM, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    txt = r["choices"][0]["message"]["content"].strip().lower()
    return "yes" if txt.startswith("y") else "no"


def main():
    global VLLM
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="temp/smoke-recheck")
    ap.add_argument("--vllm", default=VLLM, help="chat/completions 地址")
    ap.add_argument("--start", default="2026-10-27 13:15:00")
    ap.add_argument("--end", default="2026-10-27 13:45:00")
    ap.add_argument("--cams", default="1096,1168,1912")
    a = ap.parse_args()
    VLLM = a.vllm
    os.makedirs(a.out, exist_ok=True)
    cams = {c.strip() for c in a.cams.split(",") if c.strip()}

    alerts = [x for x in query_alerts(a.start, a.end) if str(x["cameraId"]) in cams]
    print(f"命中 smoking 告警 {len(alerts)} 条：")
    for x in alerts:
        print(f"  {x['eventTime']}  cam={x['cameraId']}  {x['framePath']}")

    rows = []
    for x in alerts:
        tag = x["eventTime"].replace(":", "").replace(" ", "_")
        raw = http_get(f"{PLATFORM}/evidence?path={urllib.parse.quote(x['framePath'])}")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        dets = json.loads(x.get("detectionsJson") or "[]")
        bbox = dets[0]["bbox"] if dets and dets[0].get("bbox") else locate_box(img)
        erased, red_left = erase_osd(img, bbox)
        crop = crop_person(erased, bbox) if bbox else erased

        pre = f"c{x['cameraId']}_{tag}"
        cv2.imwrite(os.path.join(a.out, f"{pre}_before.jpg"), img)
        cv2.imwrite(os.path.join(a.out, f"{pre}_erased.jpg"), erased)
        cv2.imwrite(os.path.join(a.out, f"{pre}_crop.jpg"), crop)
        ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

        t0 = datetime.now()
        verdict = ask_vlm(jpg.tobytes())
        ms = int((datetime.now() - t0).total_seconds() * 1000)
        rows.append((x["eventTime"], x["cameraId"], bbox, red_left, verdict, ms))
        print(f"  → {x['eventTime']} cam={x['cameraId']} box={bbox} 残留红像素={red_left} "
              f"VLM={verdict} ({ms}ms)  图={pre}_*.jpg")

    print("\n| 时间 | 摄像头 | bbox | 擦除后残留红 | 大模型判定 | 耗时 |")
    print("| :--- | :---: | :--- | ---: | :---: | ---: |")
    for ev, cam, bbox, red, verdict, ms in rows:
        print(f"| {ev} | {cam} | {bbox} | {red} | **{verdict}** | {ms}ms |")
    print(f"\n图片目录: {a.out}")


if __name__ == "__main__":
    main()
