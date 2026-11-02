"""低置信巡检（vlm.audit=true）：pgie@0.25 已给出框，VLM 只判人+判穿戴，**不做定位**。

节奏（2026-10-30 定稿）：
  每 audit_cycle_s（默认 2s）收各路 manager.last_low_objects（0.25~业务门）→
  排除与 ≥0.6 框重叠 → IoU+TTL 去重（同位置 audit_dedup_ttl_s=60s 内只审一次）→
  新框进线程池（默认 8 并发，目标 1 秒内消化当轮）→ 每人 3 问：
    全图+红框"是真人吗" → 头肩"戴帽吗" → 躯干"穿衣吗"，违规=真人∧未穿戴。
命中只记 vlm-audit 日志 + 落盘证据图（temp2/vlm-audit），**不推告警**。

为什么不再全屏点名：实测 VLM 的 grounding 会把配电箱/护栏框成"人"，且"判人"级
无法可靠否决自己的 hallucination（同模型自证）。定位交回小模型符合各自能力边界。
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import cv2
from loguru import logger

try:
    from server.alert import vlm_review
except ImportError:                      # 容器内平铺单测/验证时
    import vlm_review


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])
    return inter / max(1, area - inter)


def overlaps_any(bb, boxes, thr=0.3):
    return any(iou(bb, kb) > thr for kb in boxes)


def select_candidates(lows, highs, recent, now, ttl, limit=6):
    """纯函数：从低置信框里挑本轮要审的。

    lows/highs: [(bbox, conf)] / [bbox]；recent: [(bbox, ts)]（已审缓存）。
    规则：排除与 ≥0.6 重叠；与 TTL 内已审框 IoU>0.5 跳过；本轮内互相去重；按 conf 降序取前 limit。
    """
    recent_boxes = [r[0] for r in recent if now - r[1] < ttl]
    out = []
    for bbox, conf in sorted(lows, key=lambda x: -x[1]):
        if overlaps_any(bbox, highs, 0.3):
            continue
        if overlaps_any(bbox, recent_boxes, 0.5):
            continue
        if overlaps_any(bbox, [o[0] for o in out], 0.5):
            continue
        out.append((bbox, conf))
        if len(out) >= limit:
            break
    return out


def review_person(url, model, frame, bbox, timeout=2.0):
    """1 人 3 问（简单极性，违规在代码侧判定）。返回 (person, helmet, vest, violated, jpegs, ms)。"""
    t0 = time.monotonic()
    person_jpeg = vlm_review.redbox_view(frame, bbox)
    person = vlm_review.ask(url, model, person_jpeg, vlm_review.PERSON_PROMPT, timeout)
    helmet = vest = None
    head_jpeg = torso_jpeg = None
    violated = []
    if person == "yes":
        head_jpeg = vlm_review.simplejpeg_encode(vlm_review.head_crop(frame, bbox))
        helmet = vlm_review.ask(url, model, head_jpeg, vlm_review.HEAD_PROMPT, timeout)
        if helmet == "no":
            violated.append("no_helmet")
        torso_jpeg = vlm_review.simplejpeg_encode(vlm_review.torso_crop(frame, bbox))
        vest = vlm_review.ask(url, model, torso_jpeg, vlm_review.TORSO_PROMPT, timeout)
        if vest == "no":
            violated.append("no_vest")
    jpegs = {"person": person_jpeg, "head": head_jpeg, "torso": torso_jpeg}
    return person, helmet, vest, violated, jpegs, round((time.monotonic() - t0) * 1000)


def start(frame_cache, managers: dict, vlm: dict):
    """managers: {source_id: AlertManager}（取 .last_low_objects 作候选）。"""
    cycle = float(vlm.get("audit_cycle_s", 2))
    ttl = float(vlm.get("audit_dedup_ttl_s", 60))
    workers = int(vlm.get("audit_workers", 8))
    per_cycle = int(vlm.get("audit_max_per_cycle", 6))
    timeout = float(vlm.get("timeout", 2))
    url, model = vlm.get("url", ""), vlm.get("model", "")
    dump_dir = vlm.get("audit_dump_dir", "temp2/vlm-audit")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vlm-audit")
    recent: dict[int, list] = {}
    inflight = [0]
    lock = threading.Lock()

    def job(sid, camera, bbox, tag):
        try:
            entry = frame_cache.latest_with_meta(sid)
            frame = entry[0].bgr() if entry is not None else None
            if frame is None:
                return
            person, helmet, vest, violated, jpegs, ms = review_person(
                url, model, frame, bbox, timeout)
            if not violated:
                return
            if dump_dir:
                base = os.path.join(dump_dir, tag)
                for part, blob in jpegs.items():
                    if blob:
                        with open(f"{base}_{part}.jpg", "wb") as fp:
                            fp.write(blob)
                # 干净全帧（无红框，原始分辨率 1080p）：供未来重裁剪/换问句的离线 A/B 实验
                # 注意磁盘：约 0.3MB/命中；长时间开巡检时留意 temp2 体积
                cv2.imwrite(f"{base}_frame.jpg", frame)
            for rule in violated:
                wearing = helmet if rule == "no_helmet" else vest
                logger.info("vlm-audit camera={} rule={} person={} wearing={} "
                            "bbox=({},{},{},{}) → 低置信漏报候选 ({}ms, 图: {}/{}_*.jpg)",
                            camera, rule, person, wearing, *bbox, ms, dump_dir, tag)
        except Exception:
            logger.exception("vlm-audit 判定异常 camera={}", camera)
        finally:
            with lock:
                inflight[0] -= 1

    def loop():
        while True:
            t0 = time.monotonic()
            try:
                for sid, mgr in list(managers.items()):
                    if inflight[0] >= workers * 4:      # 背压：队列过深先不采集
                        break
                    lows = [(o.bbox, o.confidence) for o in (mgr.last_low_objects or [])]
                    if not lows:
                        continue
                    highs = [o.bbox for o in (mgr.last_objects or [])]
                    now = time.monotonic()
                    new = select_candidates(lows, highs, recent.get(sid, []),
                                            now, ttl, per_cycle)
                    if not new:
                        continue
                    entry = frame_cache.latest_with_meta(sid)
                    if entry is None:
                        continue
                    # 只更新去重缓存；帧在 job 里取最新（转换只发生在有候选的路）
                    recent.setdefault(sid, []).extend(new)
                    recent[sid] = [r for r in recent[sid] if now - r[1] < ttl * 2]
                    bjts = ((datetime.now(timezone.utc) + timedelta(hours=8))
                            .strftime("%m%d-%H%M%S"))
                    for k, (bbox, _conf) in enumerate(new):
                        tag = f"{bjts}_{sid}p{k}"
                        with lock:
                            inflight[0] += 1
                        pool.submit(job, sid, mgr.camera_name, bbox, tag)
            except Exception:
                logger.exception("vlm-audit 采集异常")
            time.sleep(max(0.0, cycle - (time.monotonic() - t0)))

    t = threading.Thread(target=loop, daemon=True, name="vlm-audit")
    t.start()
    return t
