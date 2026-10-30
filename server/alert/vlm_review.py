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

PRE = "这 4 张图是同一个人连续的 4 张裁剪连拍（左上、右上、左下、右下）。"
TAIL = "只回答一个词：yes 或 no。"

PROMPT = PRE + "此人是否在吸烟（手持香烟或口含香烟，无论是否点燃；夹在指间也算）？" + TAIL
# 安全帽/反光衣是静态属性，一帧即可判，无需连拍（省 ~0.5s 采样等待）。
# 两问合一：把"是否真实人员"折进同一问句——非真人框(护栏/阴影/车辆)答 no → 被排除，
# 堵死"假人必然没戴帽"的极性漏洞。三条规则极性统一：yes = 违规坐实。
# 判据按业务定义写进 prompt（灰边说明/硬壳帽 vs 布帽/反光条背心 vs 橙工作服）：
# 边界案例模型与人眼一样看不清时，显式定义就是最后的裁判标准。
HELMET_PROMPT = ("下图为一次告警目标的裁剪图（右侧或下方的灰色区域是补白，不是画面内容）。"
                 "安全帽指硬质头盔，常见黄/白/红/蓝/橙色；布帽、遮阳帽、草帽、头巾均算未佩戴。"
                 "若目标是真实人员且头上未佩戴安全帽，答 yes；"
                 "若已佩戴安全帽、或目标不是真实人员（物体/阴影/车辆等），答 no。" + TAIL)
VEST_PROMPT = ("下图为一次告警目标的裁剪图（右侧或下方的灰色区域是补白，不是画面内容）。"
               "香港工程反光衣指高可视颜色且带银灰反光条的背心；普通橙色短袖、长袖工作服均算未穿着。"
               "若目标是真实人员且未穿着香港工程反光衣，答 yes；"
               "若已穿着、或目标不是真实人员（物体/阴影/车辆等），答 no。" + TAIL)

# 规则 → (复核问题, 违规确认答案, 连拍帧数)。
REVIEWS = {
    "smoking": (PROMPT, "yes", 4),
    "no_helmet": (HELMET_PROMPT, "yes", 1),
    "no_vest": (VEST_PROMPT, "yes", 1),
}

# ---------------------------------------------------------------- v3 判问体系
# 实测教训（2026-10-30 标注考卷 56 样本）：复合条件问句("未戴→yes")会被 7B 模型
# 按直觉极性反着答（准确率仅 66.7%）。v3 全部改用**简单极性问句**，违规判定放代码里：
#   判人 = 全图+红框上下文（裁剪图孤立问人不可靠）
#   判帽 = 头肩特写（全人裁剪里帽子只有几十像素）
#   判衣 = 躯干特写
# 考卷成绩：v1 复合 66.7% / fewshot 86.1% / v3 部件问 17/17=100%（n 小，持续攒样本）。
PERSON_PROMPT = ("图中红框标记的目标是否为一名真实人员（而非护栏/阴影/车辆/机械/覆盖物）？"
                 "只回答一个词：yes 或 no。")
HEAD_PROMPT = "这是一名人员头肩部的特写。此人头上是否佩戴了安全帽？只回答一个词：yes 或 no。"
TORSO_PROMPT = "这是一名人员躯干部的特写。此人是否穿着带反光条的工程反光背心？只回答一个词：yes 或 no。"

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


def _part_crop(frame: np.ndarray, bbox, y0f: float, y1f: float, out: int = TILE) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    bh, bw = y2 - y1, x2 - x1
    sub = frame[max(0, int(y1 + y0f * bh)):max(1, int(y1 + y1f * bh)),
                max(0, int(x1 - 0.2 * bw)):min(frame.shape[1], int(x2 + 0.2 * bw))]
    if sub.size == 0:
        return np.zeros((out, out, 3), np.uint8)
    sh, sw = sub.shape[:2]
    s = out / max(sh, sw)
    rs = cv2.resize(sub, (max(1, int(sw * s)), max(1, int(sh * s))), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(rs, 0, out - rs.shape[0], 0, out - rs.shape[1],
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def head_crop(frame, bbox, out: int = TILE):
    return _part_crop(frame, bbox, -0.10, 0.45, out)      # 头肩特写


def torso_crop(frame, bbox, out: int = TILE):
    return _part_crop(frame, bbox, 0.25, 1.0, out)        # 躯干特写


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
