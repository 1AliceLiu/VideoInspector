"""Fetch the latest 100 no_helmet alarm frames and remove OSD annotations.

The platform evidence image has a red person box and a red "no_helmet" label.
We erase both, then redraw the person box in green. The cleaned frames are
later passed to helmet/utils/predict_no_helmet_person.py.

Usage:
  /home/pzq/projects/helmet/.venv/bin/python sandbox/fetch_no_helmet_frames.py \
      --out temp2/no-helmet_run --limit 100
"""
from __future__ import annotations

import argparse
import json
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PLATFORM = "http://192.168.1.24:6034/protal/ai-alarm"


def http_get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def fetch_alarms(limit: int) -> list[dict]:
    qs = urllib.parse.urlencode({"pageno": 1, "pagesize": limit, "alarmType": "no_helmet"})
    d = json.loads(http_get(f"{PLATFORM}/list?{qs}", timeout=30))
    return d["res"]["list"]


def label_rect(x1, y1, x2, y2, w, h, text="no_helmet"):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    ty = y1 - 8
    if ty - th - 8 >= 0:
        return (max(0, x1 - 6), max(0, ty - th - 10), min(w - 1, x1 + tw + 10), min(h - 1, ty + 8))
    return (max(0, x1 - 6), min(h - 1, y2 + 8), min(w - 1, x1 + tw + 10), min(h - 1, y2 + th + 16))


def clean_frame(img: np.ndarray, bboxes: list[list[int]]) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    t = 14
    for bx1, by1, bx2, by2 in bboxes:
        x1, y1, x2, y2 = [int(v) for v in (bx1, by1, bx2, by2)]
        # red box line band
        cv2.rectangle(mask, (max(0, x1 - t), max(0, y1 - t)),
                      (min(w - 1, x2 + t), min(h - 1, y1 + t)), 255, -1)
        cv2.rectangle(mask, (max(0, x1 - t), max(0, y2 - t)),
                      (min(w - 1, x2 + t), min(h - 1, y2 + t)), 255, -1)
        cv2.rectangle(mask, (max(0, x1 - t), max(0, y1 - t)),
                      (min(w - 1, x1 + t), min(h - 1, y2 + t)), 255, -1)
        cv2.rectangle(mask, (max(0, x2 - t), max(0, y1 - t)),
                      (min(w - 1, x2 + t), min(h - 1, y2 + t)), 255, -1)
        # red label above/below the box
        lx1, ly1, lx2, ly2 = label_rect(x1, y1, x2, y2, w, h)
        cv2.rectangle(mask, (lx1, ly1), (lx2, ly2), 255, -1)
    clean = cv2.inpaint(img, mask, 4, cv2.INPAINT_TELEA)
    for bx1, by1, bx2, by2 in bboxes:
        cv2.rectangle(clean, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 220, 0), 3)
    return clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="temp2/no-helmet_run")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out)
    evidence_dir = out / "evidence"
    clean_dir = out / "clean"
    if out.exists():
        shutil.rmtree(out)
    evidence_dir.mkdir(parents=True)
    clean_dir.mkdir(parents=True)

    alarms = fetch_alarms(args.limit)
    print(f"alarms: {len(alarms)}")
    meta = []
    for i, a in enumerate(alarms, 1):
        dets = json.loads(a.get("detectionsJson") or "[]")
        persons = [d for d in dets if d.get("class") == "person" and d.get("bbox")]
        bboxes = [[int(round(v)) for v in d["bbox"]] for d in persons]
        if not bboxes:
            print(f"skip {a['id']}: no person bbox")
            continue
        raw = http_get(f"{PLATFORM}/evidence?path={urllib.parse.quote(a['framePath'])}")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip {a['id']}: bad image")
            continue
        tag = f"cam{a['cameraId']}_{a['eventTime'].replace(':', '').replace(' ', '_')}_{a['id']}"
        (evidence_dir / f"{tag}.jpg").write_bytes(raw)
        clean = clean_frame(img, bboxes)
        cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
            str(clean_dir / f"{tag}.jpg")
        )
        meta.append({
            "id": a["id"],
            "eventTime": a["eventTime"],
            "cameraId": a["cameraId"],
            "framePath": a["framePath"],
            "bboxes": bboxes,
            "clean_file": f"{tag}.jpg",
        })
        if i % 10 == 0 or i == len(alarms):
            print(f"  {i}/{len(alarms)}")
    (out / "alarms.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"clean frames: {len(meta)} -> {clean_dir}")


if __name__ == "__main__":
    main()
