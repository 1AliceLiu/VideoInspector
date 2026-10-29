#!/usr/bin/env python3
"""主动打一条：用真实证据帧走一遍 VLM 复核链路（不经检测进程，零侵入）。

在 safety-detector 容器内跑（需 cv2/simplejpeg，且容器网络可达 vllm 服务）：
    docker cp deploy/alerts/671/xxx.jpg safety-detector:/tmp/vi-check/
    docker exec -w /tmp/vi-check safety-detector python3 probe_vllm.py xxx.jpg \
        --bbox 0.775,0.27,0.84,0.47
不传 --bbox 则裁画面中心 60%。判定预期与图像内容一致（无人吸烟 → no）。
"""

import argparse
import sys
import time

import cv2
from loguru import logger

from server.alert import vlm_review

URL = "http://vllm:8013/v1/chat/completions"   # 宿主机上跑则改 http://127.0.0.1:8013/...
MODEL = "/model/Qwen2.5-VL-7B-Instruct-AWQ"

ap = argparse.ArgumentParser()
ap.add_argument("jpg", nargs="+", help="一个或多个证据帧")
ap.add_argument("--bbox", default=None, help="person 框比例 x1,y1,x2,y2 (0~1)")
a = ap.parse_args()

for path in a.jpg:
    img = cv2.imread(path, cv2.IMREAD_COLOR)   # 强制BGR三通道：历史告警图有灰度编码的
    if img is None:
        logger.error("读不到图: {}", path)
        continue
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (map(float, a.bbox.split(",")) if a.bbox
                      else (w * 0.2, h * 0.2, w * 0.8, h * 0.8))
    bbox = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))
    crops = [vlm_review.crop_person(img, bbox)] * 4   # 单图重复4格：只验问答链路
    t0 = time.monotonic()
    verdict = vlm_review.ask(URL, MODEL, vlm_review.build_grid(crops))
    dt = round((time.monotonic() - t0) * 1000)
    print(f"{path}  bbox={bbox}  verdict={verdict}  total={dt}ms", flush=True)
sys.exit(0)
