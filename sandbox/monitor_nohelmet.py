#!/usr/bin/env python3
"""监控指定路的 no-helmet 帧并抓取。生产口径：person(yolo26n@640)>=0.6，head(helmet检测器)>=0.5 且 head 中心落在 person 框内。

用法（容器内）：CUDA_VISIBLE_DEVICES=2 python3 monitor_nohelmet.py --config config.yaml --id 1168
                     --person /tmp/nh/yolo26n.pt --helmet /tmp/nh/helm.pt --duration 300 --out out
"""
import argparse
import os
import time

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "ERROR")
import cv2
import eval_person as ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--id", default="1168")
    ap.add_argument("--person", default="yolo26n.pt")
    ap.add_argument("--helmet", default="helm.pt")
    ap.add_argument("--person-conf", type=float, default=0.6, help="业务门（线上现值0.6）")
    ap.add_argument("--head-conf", type=float, default=0.5, help="attribute_threshold 口径")
    ap.add_argument("--stride", type=int, default=10, help="每N帧检一次（≈1秒）")
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cam = next(c for c in ep.load_cameras(a.config) if c["id"] == str(a.id))
    from ultralytics import YOLO
    pmodel, hmodel = YOLO(a.person), YOLO(a.helmet)
    print("helmet classes:", hmodel.names, flush=True)

    url = cam["rtsp_url"]
    t_end = time.monotonic() + a.duration
    n = hits = 0
    while time.monotonic() < t_end:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG,
                               [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000,
                                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("打开失败，2s 后重试", flush=True)
            time.sleep(2)
            continue
        while time.monotonic() < t_end:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % a.stride:
                continue
            pdet = pmodel(frame, conf=a.person_conf, classes=[0], verbose=False)[0]
            if len(pdet.boxes) == 0:
                continue
            hdet = hmodel(frame, conf=a.head_conf, verbose=False)[0]
            viol = []
            for ci, hc, hb in zip(hdet.boxes.cls, hdet.boxes.conf, hdet.boxes.xyxy):
                if hmodel.names[int(ci)] != "head":
                    continue
                x1, y1, x2, y2 = map(int, hb)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                for pb in pdet.boxes.xyxy:
                    px1, py1, px2, py2 = map(int, pb)
                    if px1 <= cx <= px2 and py1 <= cy <= py2:
                        viol.append((float(hc), (x1, y1, x2, y2), (px1, py1, px2, py2)))
                        break
            if not viol:
                continue
            hits += 1
            vis = frame.copy()
            for _, (hx1, hy1, hx2, hy2), (px1, py1, px2, py2) in viol:
                cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 210, 0), 2)
                cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), (0, 0, 235), 2)
            ts = time.strftime("%H%M%S")
            f = os.path.join(a.out, f"cam{cam['id']}_{ts}_n{hits:03d}.jpg")
            cv2.imwrite(f, vis)
            print(f"no-helmet 抓取 {f} persons={len(pdet.boxes)} heads={len(viol)}", flush=True)
        cap.release()
    print(f"\n===== 读{n}帧 检{n // a.stride}次 命中{hits}帧 -> {a.out} =====")


if __name__ == "__main__":
    main()
