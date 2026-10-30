"""VLM 主动巡检（config vlm.audit=true 启用）：GPU 空闲期让大模型"巡逻"全部摄像头。

每路每 audit_interval_s 秒：取最新证据帧 → 大模型全屏点名所有真实人员（grounding
归一化框）→ 与检测器本帧上报的人框比对，**不重叠的 = 检测器漏掉的可疑目标** →
用 v3 部件问句（全图红框判真人 + 头/躯干特写判穿戴）二次确认 → 两级都过才记
`vlm-audit` 日志（只观察不推告警），作为召回审计与后续补报的数据源。

为什么这样设计：全屏点名本身会幻觉（实测把 4 人数成 12），但"点名 → 裁剪复核"
两级串联后，幻觉框会在复核级被"非真实人员"挡掉（s02 半个人案例则能被捞回）。
对检测流零影响：只读帧缓存 + 发 HTTP，独立 daemon 线程，速率受 interval 限制。
"""

import re
import threading
import time

import cv2
from loguru import logger

try:
    from server.alert import vlm_review
except ImportError:                      # 容器内平铺单测/验证时
    import vlm_review

SCENE_PROMPT = ("这是工地监控画面。列出画面中所有真实人员（护栏/阴影/车辆/机械/覆盖物"
                "不是人）。每人输出一行，格式严格为：人|x1,y1,x2,y2（坐标0-1000归一化）。"
                "只输出行，不要其他文字。")


def parse_scene(text: str, w: int, h: int):
    out = []
    for m in re.finditer(r"(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})", text):
        x1, y1, x2, y2 = (int(v) for v in m.groups())
        if x2 > x1 and y2 > y1:
            out.append((x1 * w / 1000, y1 * h / 1000, x2 * w / 1000, y2 * h / 1000))
    return out


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])
    return inter / max(1, area - inter)


def audit_frame(frame, known_boxes, url, model, timeout=5.0):
    """返回检测器漏报且经二级确认的可疑违规 [(bbox, rule, person, wearing)]。

    幻觉控制：点名框按面积取前 3（真人通常比幻觉框大）；判人只问一次，
    非真人立即跳过（实测幻觉框 100% 死在这一级）。
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (960, 540))
    jpeg = cv2.imencode(".jpg", small)[1].tobytes()
    text = vlm_review.ask_text(url, model, jpeg, SCENE_PROMPT, max_tokens=220,
                               timeout=timeout)
    claims = parse_scene(text or "", w, h)
    claims.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    hits = []
    for bb in claims[:3]:
        if any(iou(bb, kb) > 0.3 for kb in known_boxes):
            continue                                   # 检测器已看到，不重复审计
        person = vlm_review.ask(url, model, vlm_review.redbox_view(frame, bb),
                                vlm_review.PERSON_PROMPT, timeout)
        if person != "yes":
            continue
        for rule, prompt, cropf in (
                ("no_helmet", vlm_review.HEAD_PROMPT, vlm_review.head_crop),
                ("no_vest", vlm_review.TORSO_PROMPT, vlm_review.torso_crop)):
            wearing = vlm_review.ask(url, model,
                                     vlm_review.simplejpeg_encode(cropf(frame, bb)),
                                     prompt, timeout)
            if wearing == "no":
                hits.append((bb, rule, person, wearing))
    return hits


def start(frame_cache, managers: dict, vlm: dict):
    """managers: {source_id: AlertManager}（取 .last_objects 作检测器本帧名单）。"""
    interval = float(vlm.get("audit_interval_s", 90))
    url, model = vlm.get("url", ""), vlm.get("model", "")

    def loop():
        sids = list(managers)
        i = 0
        while True:
            sid = sids[i % len(sids)]
            i += 1
            time.sleep(interval / max(1, len(sids)))     # 均摊：全轮一遍≈interval
            try:
                entry = frame_cache.latest_with_meta(sid)
                if entry is None:
                    continue
                frame = entry[0].bgr()
                known = [o.bbox for o in (managers[sid].last_objects or [])]
                for bb, rule, person, wearing in audit_frame(frame, known, url, model):
                    logger.info("vlm-audit camera={} rule={} person={} wearing={} "
                                "bbox=({:.0f},{:.0f},{:.0f},{:.0f}) → 检测器漏报候选",
                                managers[sid].camera_name, rule, person, wearing, *bb)
            except Exception:
                logger.exception("vlm-audit 巡检异常 source={}", sid)

    t = threading.Thread(target=loop, daemon=True, name="vlm-audit")
    t.start()
    return t
