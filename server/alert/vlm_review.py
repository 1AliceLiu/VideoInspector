"""告警的视觉大模型二级复核：连续 4 帧人像裁剪 → 2×2 宫格图 → yes/no。

定位（配置见 deploy/config.yaml vlm 节）：小模型仍走既有 DeepStream 链实时初筛；
状态机触发告警后**不直接推 webhook**，本模块在证据帧缓存里轮询触发后的新帧（采样与
HTTP 全在告警 daemon 线程，不碰检测流线程），按触发时刻的 person bbox 裁剪连拍人像，
拼成 2×2 宫格图（一次调用覆盖 ~0.5s 连拍，固定开销只付一遍），问 OpenAI 兼容接口
（如 vLLM 部署的 Qwen2.5-VL，max_tokens=1）。每规则的复核问题与违规确认方向见 REVIEWS。

判定语义（shadow=true 观察模式下不拦告警、payload 不附字段，结论只进日志）：
  verdict == 确认方向 → 正常推送（payload 附 vlm.verdict）
  verdict == 相反方向 → 丢弃（误报被大模型排除），仅记日志
  不可用/无帧 → fail_open（默认 true）照常推送，verdict=skipped —— 宁误报不漏报

配置键 deploy/config.yaml vlm 节：enabled/shadow/url/model/rules/
sample_ms/deadline_ms/pad（bbox 外扩比例，默认 0.15）/fail_open/timeout；
每规则的问题/确认方向/连拍帧数在 REVIEWS（吸烟 4 帧，安全帽/反光衣 1 帧）。
rules 只支持 REVIEWS 中的键名，未定义的规则直接推送并告警（问不对问题不如不问）。
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
TORSO_TILE = 512  # 躯干特写用 512：反光条是细横条，256px 下低对比条带看不见（实测）

PRE = ("These 4 images are 4 consecutive cropped frames of the same person "
       "(top-left, top-right, bottom-left, bottom-right). ")
TAIL = "Answer with exactly one word: yes or no."

PROMPT = PRE + ("Is this person smoking (holding a cigarette or having one in their mouth, "
                "whether lit or not; holding it between the fingers also counts)? ") + TAIL
# 安全帽/反光衣是静态属性，一帧即可判，无需连拍（省 ~0.5s 采样等待）。
# 两问合一：把"是否真实人员"折进同一问句——非真人框(护栏/阴影/车辆)答 no → 被排除，
# 规则 → (复核问题, 违规确认答案, 连拍帧数)。
# 说明：PPE（no_helmet/no_vest）已改走 v3 部件问句（见下方 HEAD/TORSO_PROMPT + RULE_MODE），
# 不再使用整人裁剪图；REVIEWS 仅保留吸烟（burst）供旁路脚本复用。
REVIEWS = {
    "smoking": (PROMPT, "yes", 4),
}

# ---------------------------------------------------------------- v3 判问体系
# 实测教训（2026-10-30 标注考卷 56 样本）：复合条件问句("未戴→yes")会被 7B 模型
# 按直觉极性反着答（准确率仅 66.7%）。v3 全部改用**简单极性问句**，违规判定放代码里：
#   判人 = 全图+红框上下文（裁剪图孤立问人不可靠）
#   判帽 = 头肩特写（全人裁剪里帽子只有几十像素）
#   判衣 = 躯干特写
# 考卷成绩：v1 复合 66.7% / fewshot 86.1% / v3 部件问 17/17=100%（n 小，持续攒样本）。
PERSON_PROMPT = ("Is the target marked by the red box in the image a real person "
                 "(rather than a guardrail, shadow, vehicle, machinery, or covering)? "
                 "Answer with exactly one word: yes or no.")
# PPE 问句必须用「如果…回答 yes，否则 no」的对齐句式：
# 实测"是否佩戴…？"句式系统性偏否（头部 19/19 误答未戴、躯干 20/20 误答未穿，
# 而自由描述明明说出"黄色安全帽/反光背心"）；对齐句式 3/3 正确、分布合理。
# 注意：判人问句不要用对齐句式（模型会复述题目而非作答，实测输出"图"）。
# 实测（2026-10-30，40 张真实头肩图）：带业务判据的详细版 33/33，
# 简短版 32/33——错例正是"草帽"（详细版的"草帽不算安全帽"条款救回）。
HEAD_PROMPT = ("This is a close-up of a person's head and shoulders. A safety helmet means "
               "a hard hat (commonly yellow, white, red, blue, or orange); cloth caps, sun hats, "
               "straw hats, and headscarves do not count as safety helmets by themselves. Note: "
               "on construction sites, sun visors, sun capes, and neck covers are often attached "
               "to safety helmets, or a sun hat is worn over a safety helmet. As long as the hard "
               "shell is visible (hemispherical crown, headband, hard edge under the brim), it "
               "counts as wearing a safety helmet; this applies equally to front, side, and rear "
               "views. If this person is wearing a safety helmet on their head, answer yes; "
               "otherwise answer no.")
TORSO_PROMPT = ("This is a close-up of a person's torso (from the shoulders down to the thighs). "
                "A Hong Kong construction reflective vest means reflective workwear with reflective "
                "stripes: any color shade is acceptable, and the style may be a vest, waistcoat, "
                "long-sleeve or short-sleeve; ordinary clothing without any reflective stripes "
                "does not count. If this person is wearing reflective clothing with reflective "
                "stripes, answer yes; otherwise answer no.")

RULE_MODE = {   # 规则 → (复核方式, 违规确认值, 连拍帧数)
    "smoking": ("burst", "yes", 4),
    "no_helmet": ("ppe", "yes", 1),
    "no_vest": ("ppe", "yes", 1),
}


def redbox_view(frame: np.ndarray, bbox, w: int = 960, h: int = 540) -> bytes:
    """整帧缩到 w×h 并画红框标记候选目标——给"是否真人"问句提供场景上下文。"""
    fh, fw = frame.shape[:2]
    sx, sy = w / fw, h / fh
    vis = cv2.resize(frame, (w, h))
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (0, 0, 255), 3)
    return cv2.imencode(".jpg", vis)[1].tobytes()


def _part_crop(frame: np.ndarray, bbox, y0f: float, y1f: float, out: int = TILE,
               xpad: float = 0.2) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    bh, bw = y2 - y1, x2 - x1
    sub = frame[max(0, int(y1 + y0f * bh)):max(1, int(y1 + y1f * bh)),
                max(0, int(x1 - xpad * bw)):min(frame.shape[1], int(x2 + xpad * bw))]
    if sub.size == 0:
        return np.zeros((out, out, 3), np.uint8)
    sh, sw = sub.shape[:2]
    s = out / max(sh, sw)
    interp = cv2.INTER_LANCZOS4 if s >= 1 else cv2.INTER_AREA   # 放大用 LANCZOS 保边缘，缩小用 AREA
    rs = cv2.resize(sub, (max(1, int(sw * s)), max(1, int(sh * s))), interpolation=interp)
    return cv2.copyMakeBorder(rs, 0, out - rs.shape[0], 0, out - rs.shape[1],
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def head_crop(frame, bbox, out: int = TILE):
    # 裁剪几何保持纵向原值（-10%→45%）；横向按业务要求收窄（xpad 0.2→0）
    return _part_crop(frame, bbox, -0.10, 0.45, out, xpad=0.0)


def torso_crop(frame, bbox, out: int = TORSO_TILE):
    # 纵向 10%→100%（含肩/上胸）；横向收窄（xpad=0）；512px 让反光细条可见
    return _part_crop(frame, bbox, 0.10, 1.0, out, xpad=0.0)


def judge_ppe(url: str, model: str, frame: np.ndarray, bbox, rule: str,
              timeout: float = 2.0):
    """v3 单人判定：全图红框问真人 + 部位特写问穿戴。
    返回 (person, wearing, violated, ms)；violated = 真人且未穿戴。"""
    t0 = time.monotonic()
    person = ask(url, model, redbox_view(frame, bbox), PERSON_PROMPT, timeout)
    wearing = ask(url, model,
                  simplejpeg_encode(head_crop(frame, bbox)) if rule == "no_helmet"
                  else simplejpeg_encode(torso_crop(frame, bbox)),
                  HEAD_PROMPT if rule == "no_helmet" else TORSO_PROMPT, timeout)
    violated = person == "yes" and wearing == "no"
    return person, wearing, violated, round((time.monotonic() - t0) * 1000)


def simplejpeg_encode(arr: np.ndarray) -> bytes:
    return simplejpeg.encode_jpeg(arr, quality=90, colorspace="BGR")


def crop_person(frame: np.ndarray, bbox, out: int = TILE, pad: float = 0.15):
    """原始帧按 bbox 裁人（向外扩 pad 防截掉烟头），**等比缩放**贴到 out×out 灰底。

    退化/越界框返回 None（调用方跳过这一帧，等新帧号再采）。
    必须保持宽高比：直接 resize 成正方形会把蹲伏/宽扁框的人拉成"压缩饼"，
    实测导致 Qwen2.5-VL 对"是否佩戴安全帽"的 yes/no 绑定错乱（同字节重放 5/5 判错，
    等比重裁 5/5 判对）。
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
    x2, y2 = min(w, int(x2 + px)), min(h, int(y2 + py))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    sub = frame[y1:y2, x1:x2]
    sh, sw = sub.shape[:2]
    s = out / max(sh, sw)
    rs = cv2.resize(sub, (max(1, int(sw * s)), max(1, int(sh * s))),
                    interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(rs, 0, out - rs.shape[0], 0, out - rs.shape[1],
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def build_grid(crops: list) -> bytes:
    """单帧直通；≤4 张裁剪拼 2×2 宫格 JPEG（不足 4 张用最后一张补齐）。"""
    if len(crops) == 1:
        return simplejpeg.encode_jpeg(crops[0], quality=90, colorspace="BGR")
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
                if len(crops) >= frames:
                    break          # 采满立刻走，不空等最后一个 sleep（单帧规则 0 等待）
        time.sleep(sample_s)
    return crops


def ask_text(url: str, model: str, jpeg: bytes, prompt: str,
             max_tokens: int = 1, timeout: float = 5.0, temperature: float = 0.0):
    """宫格/整图问 OpenAI 兼容 /v1/chat/completions，返回原始文本；失败返回 None。"""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                       + base64.b64encode(jpeg).decode("ascii")}},
        ]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            answer = json.loads(resp.read())
        return str(answer["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        logger.warning("vlm-review 请求失败 url={} err={}", url, exc)
        return None


def ask(url: str, model: str, jpeg: bytes, prompt: str = PROMPT, timeout: float = 5.0):
    """yes/no 单问封装：返回 "yes"/"no"，请求失败或响应异常返回 None（调用方按 fail_open 处置）。"""
    text = (ask_text(url, model, jpeg, prompt, timeout=timeout) or "").lower()
    if not text:
        return None
    return "yes" if text.startswith("y") else "no"
