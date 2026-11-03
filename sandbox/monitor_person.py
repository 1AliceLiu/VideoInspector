#!/usr/bin/env python3
"""持续监测：并发拉取 config.yaml 全部 RTSP 流 N 秒，yolo 检 person，按业务门梯度落盘。

用法（容器内）：CUDA_VISIBLE_DEVICES=2 python3 monitor_person.py --config config.yaml
                     --model /workspace/yolo26s.pt --imgsz 1280 --duration 300 --stride 10
"""
import argparse
import csv
import os
import threading
import queue
import time
from collections import defaultdict

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "ERROR")
import cv2
import numpy as np
import eval_person as ep

FLOOR = 0.25    # 引擎门：低于它的框生产链路看不到，不落盘


def worker(cam, q, stop, stride):
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000]
    n = fails = 0
    while not stop.is_set() and fails < 3:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(cam["rtsp_url"], cv2.CAP_FFMPEG, params)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            fails += 1
            time.sleep(2)
            continue
        n = 0
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            if n % stride == 0:
                # 只回传 jpeg 编码字节，避开主线程长时间持有数千帧 raw 的内存峰值
                q.put((cam["id"], cam["name"], time.strftime("%H%M%S"),
                       cv2.imencode(".jpg", frame)[1].tobytes()))
        cap.release()
        if stop.is_set():
            break
        fails += 1
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default="/workspace/yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--stride", type=int, default=10, help="每 N 帧检一次（约1秒）")
    ap.add_argument("--duration", type=int, default=300, help="监测秒数")
    ap.add_argument("--out", default="out", help="输出根目录（下建 conf06/055/05/045/04）")
    ap.add_argument("--gates", default="0.6,0.55,0.5,0.45,0.4")
    a = ap.parse_args()

    cams = ep.load_cameras(a.config)
    gates = [float(x) for x in a.gates.split(",")]
    from ultralytics import YOLO
    model = YOLO(a.model)

    q = queue.Queue(maxsize=1000)
    stop = threading.Event()
    ths = [threading.Thread(target=worker, args=(c, q, stop, a.stride), daemon=True) for c in cams]
    for t in ths:
        t.start()
    print(f"拉流 {len(cams)} 路，监测 {a.duration}s，imgsz={a.imgsz}", flush=True)

    t_end = time.monotonic() + a.duration
    n_img = n_hit = 0
    stats = {g: [0, 0] for g in gates}    # gate -> [图片数, 框数]
    dirs = {g: os.path.join(a.out, "conf" + str(g).split(".")[1].rjust(2, "0")) for g in gates}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    handles = {g: open(os.path.join(dirs[g], "predictions.csv"), "w", newline="", encoding="utf-8-sig") for g in gates}
    writers = {g: csv.writer(handles[g]) for g in gates}
    for g in gates:
        writers[g].writerow(["图文件名", "摄像头ID", "conf", "bbox"])

    while time.monotonic() < t_end:
        try:
            cid, cname, ts, buf = q.get(timeout=2)
        except queue.Empty:
            continue
        n_img += 1
        frame = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        det = model(frame, conf=FLOOR, imgsz=a.imgsz, classes=[0], verbose=False)[0]
        boxes = sorted(((round(float(b.conf), 2), tuple(map(int, b.xyxy[0]))) for b in det.boxes), key=lambda x: -x[0])
        if boxes:
            n_hit += 1
            for g in gates:
                sel = [b for b in boxes if b[0] >= g]
                if not sel:
                    continue
                img_name = f"cam{cid}_{ts}_g{int(g * 100)}.jpg"
                vis, _ = ep.draw_annotated_frame(cv2, frame, sel, cid, cname, 0)
                cv2.imwrite(os.path.join(dirs[g], img_name), vis)
                for conf, (x1, y1, x2, y2) in sel:
                    writers[g].writerow([img_name, cid, f"{conf:.2f}", f"{x1},{y1},{x2},{y2}"])
                    stats[g][1] += 1
                stats[g][0] += 1
        if n_img % 200 == 0:
            print(f"已检 {n_img} 帧，命中 {n_hit}，剩余 {int(t_end - time.monotonic())}s", flush=True)

    stop.set()
    for h in handles.values():
        h.close()
    print(f"\n===== 共检 {n_img} 帧，{n_hit} 帧含 person(>={FLOOR:g}) =====")
    for g in gates:
        print(f"门 {g:g}: {stats[g][0]} 张图 / {stats[g][1]} 个框 -> {dirs[g]}")


if __name__ == "__main__":
    main()
