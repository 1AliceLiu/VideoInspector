"""吸烟告警的视觉大模型二级复核：连续 4 帧人像裁剪 → 2×2 宫格图 → yes/no。

定位（配置见 deploy/config.yaml vlm 节）：smoke 小模型仍走既有 DeepStream 链实时初筛；
状态机触发告警后**不直接推 webhook**，本模块在证据帧缓存里轮询触发后的新帧（采样与
HTTP 全在告警 daemon 线程，不碰检测流线程），按触发时刻的 person bbox 裁剪连拍人像，
拼成 2×2 宫格图（一次调用覆盖 ~0.5s 连拍，固定开销只付一遍），问 OpenAI 兼容接口
（如 vLLM 部署的 Qwen2.5-VL，max_tokens=1）「此人是否在吸烟」。

判定语义（shadow=true 观察模式下不拦告警、payload 不附字段，结论只进日志）：
  yes        → 正常推送（payload 附 vlm.verdict=yes）
  no         → 丢弃，仅记日志
  不可用/无帧 → fail_open（默认 true）照常推送，verdict=skipped —— 宁误报不漏报

配置键 deploy/config.yaml vlm 节：enabled/shadow/url/model/rules/frames/sample_ms/
deadline_ms/pad（bbox 外扩比例，默认 0.15）/fail_open/timeout/prompt，见 manager.py _vlm_review。
"""

import base64
import json
import time
import urllib.request

import cv2
import numpy as np
import simplejpeg
from loguru import logger

TILE = 256  # 宫格单格边长 px：4 格拼 512×512，视觉 token 少，单张 L20 一次调用 ~0.1s

PROMPT = (
    "这 4 张图是同一个人连续的 4 张裁剪连拍（左上、右上、左下、右下）。"
    "此人是否在吸烟（手持香烟或口含香烟，无论是否点燃）？只回答一个词：yes 或 no。"
)


def crop_person(frame: np.ndarray, bbox, out: int = TILE, pad: float = 0.15):
    """原始帧按 bbox 裁人（向外扩 pad 防截掉烟头），resize 到 out×out。

    退化/越界框返回 None（调用方跳过这一帧，等新帧号再采）。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], (out, out), interpolation=cv2.INTER_AREA)


def build_grid(crops: list) -> bytes:
    """≤4 张裁剪拼 2×2 宫格 JPEG；不足 4 张用最后一张补齐（大模型视角=一次连拍）。"""
    tiles = crops[:4]
    filler = tiles[-1] if tiles else np.zeros((TILE, TILE, 3), np.uint8)
    while len(tiles) < 4:
        tiles.append(filler)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    return simplejpeg.encode_jpeg(grid, quality=90, colorspace="BGR")


def collect_frames(frame_cache, source_id, bbox,
                   frames: int = 4, sample_s: float = 0.12, deadline_s: float = 1.5,
                   pad: float = 0.15):
    """触发时刻后从证据缓存轮询**新帧号**并裁剪人员，最多 frames 张 / deadline_s 秒。

    证据缓存每路只存最新一帧（appsink 覆盖写），采样间隔 ≥ 帧间隔即可拿到连拍；
    断流时帧号不变，自然停止累计，由 deadline 兜底。pad 透传给 crop_person。
    """
    crops, last_fn = [], None
    end = time.monotonic() + deadline_s
    while len(crops) < frames and time.monotonic() < end:
        entry = frame_cache.latest_with_meta(source_id)
        if entry is not None and entry[1] != last_fn:
            last_fn = entry[1]
            crop = crop_person(entry[0].bgr(), bbox, pad=pad)
            if crop is not None:
                crops.append(crop)
        time.sleep(sample_s)
    return crops


def ask(url: str, model: str, jpeg: bytes, prompt: str = PROMPT, timeout: float = 5.0):
    """宫格图问 OpenAI 兼容 /v1/chat/completions（max_tokens=1）。返回 "yes"/"no"，
    请求失败或响应异常返回 None（调用方按 fail_open 处置）。"""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                       + base64.b64encode(jpeg).decode("ascii")}},
        ]}],
        "max_tokens": 1,
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            answer = json.loads(resp.read())
        text = str(answer["choices"][0]["message"]["content"]).strip().lower()
    except Exception as exc:
        logger.warning("vlm-review 请求失败 url={} err={}", url, exc)
        return None
    if not text:
        return None
    return "yes" if text.startswith("y") else "no"
