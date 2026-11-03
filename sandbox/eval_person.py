#!/usr/bin/env python3
"""yolo26n 检人准确度旁路评测工具（沙箱运行：抽帧预画框 + 人工零拉框核对 + 算分）。

用法：
  1. 抽帧预画框（在 helmet-dev 容器 /tmp/sandbox 内跑，用空闲 GPU2，零生产影响）：
     python3 eval_person.py sample --config config.yaml --model /tmp/p.pt --out /tmp/sandbox --gate 0.3

  2. 人工核对 labels.csv 后算分（宿主机或容器内均可直接跑，仅依赖标准库，--gate 须与 sample 一致）：
     python3 sandbox/eval_person.py score --dir sandbox --gate 0.3

  3. 自检（验证算分逻辑边界用例）：
     python3 sandbox/eval_person.py selftest
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

GATE = 0.30          # 业务门 person_conf_threshold（sample/score 用 --gate 覆盖）
ENGINE_GATE = 0.25   # pgie 引擎门（pre-cluster-threshold）：低于它的框生产链路根本看不到
FLOOR = 0.15         # 采样下限：引擎门外仅观察用（红桶）
BUCKET_NONE = "无检出框"


def bucket_names():
    return (f">={GATE:g}(绿-现门内)",
            f"{ENGINE_GATE:g}~{GATE:g}(黄-引擎门内)",
            f"{FLOOR:g}~{ENGINE_GATE:g}(红-引擎门外)")


BUCKET_COLORS = [(0, 210, 0), (0, 210, 255), (0, 0, 235)]    # 绿/黄/红（按桶下标）


def bucket_idx(c):
    return 0 if c >= GATE else (1 if c >= ENGINE_GATE else 2)

LABEL_HEADERS = [
    "图文件名",
    "摄像头ID",
    "摄像头名称",
    "北京时间",
    "框编号",
    "conf",
    "conf分桶",
    "bbox",
    "是真人(1=是/0=假框/?=小像素忽略)",
    "框完整(1=完整/0=截断半身)",
    "未画框漏检真人数(连红黄绿框都没套住的可看清穿戴真人)",
]

STATUS_HEADERS = [
    "序号",
    "摄像头ID",
    "摄像头名称",
    "rtsp_url",
    "状态",
    "连接方式",
    "采集帧数",
    "检出总框数(>=0.15)",
    "绿框数(现门内)",
    "黄框数(引擎门~业务门)",
    "红框数(引擎门外)",
    "耗时_s",
]


def conf_bucket(c):
    return bucket_names()[bucket_idx(c)]


def bj_now():
    return (datetime.now(timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def load_cameras(config_path):
    import yaml
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cams = []
    for c in cfg.get("cameras", []):
        if not c.get("enabled", True):
            continue
        cams.append({
            "id": str(c.get("id", "")).strip(),
            "name": str(c.get("name", "")).strip(),
            "rtsp_url": str(c.get("rtsp_url", "")).strip(),
        })
    return cams


def draw_annotated_frame(cv2, frame, boxes, cam_id, cam_name, f_idx):
    """在全图上画分桶色框与编号，标签置于框外避免遮挡人头/反光衣。"""
    vis = frame.copy()
    h, w = vis.shape[:2]
    counts = [0, 0, 0]    # 绿/黄/红
    for k, (conf, (x1, y1, x2, y2)) in enumerate(boxes):
        idx = bucket_idx(conf)
        counts[idx] += 1
        col = BUCKET_COLORS[idx]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        txt = f"#{k} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y1 - 6 if y1 > th + 10 else min(h - 4, y2 + th + 8)
        tx = max(2, min(x1, w - tw - 4))
        cv2.rectangle(vis, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 4), (20, 20, 20), -1)
        cv2.putText(vis, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    banner = (f"cam={cam_id} ({cam_name}) f={f_idx} | "
              f"boxes={len(boxes)} [G>={GATE:g}:{counts[0]} "
              f"Y{ENGINE_GATE:g}-{GATE:g}:{counts[1]} R<{ENGINE_GATE:g}:{counts[2]}]")
    cv2.rectangle(vis, (0, 0), (min(w, 920), 32), (20, 20, 20), -1)
    cv2.putText(vis, banner, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
    return vis, counts


def pull_stream_frames(cv2, url, per_cam, stride, max_read):
    """先用 TCP 拉流（同生产 rtsp_protocol=4），失败则自动回退默认协议；
    连接超时 5s（离线流快速失败），首帧等待 15s（跨过 H.265 长 GOP 等 IDR 帧），首帧后读超时缩至 3s。
    """
    params = [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000,
    ]
    for mode, opts in (("tcp", "rtsp_transport;tcp"), ("auto", "")):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            continue
        picked = []
        last_good = None
        t_first = None
        for n in range(1, max_read + 1):
            if t_first is not None and time.monotonic() - t_first > 6.0 and picked:
                break
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                break
            if t_first is None:
                t_first = time.monotonic()
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000)
            last_good = frame
            if n == 1 or n % stride == 0:
                picked.append(frame.copy())
                if len(picked) >= per_cam:
                    break
        cap.release()
        if not picked and last_good is not None:
            picked.append(last_good)
        if picked:
            return picked, mode
    return [], "-"


def cmd_sample(args):
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "ERROR")
    import cv2
    from ultralytics import YOLO

    cams = load_cameras(args.config)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    model = YOLO(args.model)

    labels_path = os.path.join(args.out, "labels.csv")
    status_path = os.path.join(args.out, "stream_status.csv")

    prev_labels = defaultdict(list)
    prev_status = {}
    if (args.only or args.retry_offline) and os.path.exists(labels_path) and os.path.exists(status_path):
        with open(labels_path, encoding="utf-8-sig") as f:
            for r in csv.reader(f):
                if r and r[0] != LABEL_HEADERS[0]:
                    prev_labels[r[1]].append(r)
        with open(status_path, encoding="utf-8-sig") as f:
            for r in csv.reader(f):
                if r and r[0] != STATUS_HEADERS[0]:
                    prev_status[r[1]] = r

    only_set = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None

    new_labels = []
    new_status = []
    online = 0
    total_frames = 0
    total_boxes = 0

    for idx, cam in enumerate(cams, 1):
        cid, cname, url = cam["id"], cam["name"], cam["rtsp_url"]
        skip = False
        if only_set is not None and cid not in only_set:
            skip = True
        elif args.retry_offline and prev_status.get(cid, [None]*5)[4] == "在线已采":
            skip = True

        if skip and cid in prev_status:
            st_row = prev_status[cid]
            st_row[0] = idx
            new_status.append(st_row)
            new_labels.extend(prev_labels.get(cid, []))
            if st_row[4] == "在线已采":
                online += 1
                total_frames += int(st_row[6])
                total_boxes += int(st_row[7])
            continue

        t0 = time.monotonic()
        frames, mode = pull_stream_frames(
            cv2, url, args.per_cam, args.stride, args.per_cam * args.stride + 5
        )
        dt = round(time.monotonic() - t0, 1)
        if not frames:
            new_status.append([idx, cid, cname, url, "离线失败", "-", 0, 0, 0, 0, 0, dt])
            print(f"[{idx:02d}/{len(cams)}] ✗ cam={cid:4s} ({cname}) 拉流失败 ({dt}s)", flush=True)
            continue

        online += 1
        cam_counts = [0, 0, 0]
        cam_boxes = 0

        for old_f in os.listdir(img_dir):
            if old_f.startswith(f"cam{cid}_f") and old_f.endswith(".jpg"):
                os.remove(os.path.join(img_dir, old_f))

        for f_i, frame in enumerate(frames, 1):
            total_frames += 1
            ts = bj_now()
            det = model(frame, conf=args.conf, imgsz=args.imgsz, classes=[0], verbose=False)[0]
            boxes = sorted(
                ((round(float(b.conf), 2), tuple(map(int, b.xyxy[0]))) for b in det.boxes),
                key=lambda x: -x[0],
            )
            img_name = f"cam{cid}_f{f_i:02d}.jpg"
            vis, c_bkt = draw_annotated_frame(cv2, frame, boxes, cid, cname, f_i)
            cv2.imwrite(os.path.join(img_dir, img_name), vis)

            for i in range(3):
                cam_counts[i] += c_bkt[i]
            cam_boxes += len(boxes)
            total_boxes += len(boxes)

            if not boxes:
                new_labels.append([img_name, cid, cname, ts, "-", "-", BUCKET_NONE, "-", "-", "-", "0"])
            else:
                for b_i, (conf, (x1, y1, x2, y2)) in enumerate(boxes):
                    new_labels.append([
                        img_name,
                        cid,
                        cname,
                        ts,
                        f"#{b_i}",
                        f"{conf:.2f}",
                        conf_bucket(conf),
                        f"{x1},{y1},{x2},{y2}",
                        "1",
                        "1",
                        "0" if b_i == 0 else "",
                    ])
        new_status.append([
            idx, cid, cname, url, "在线已采", mode, len(frames), cam_boxes,
            cam_counts[0], cam_counts[1], cam_counts[2], dt
        ])
        print(f"[{idx:02d}/{len(cams)}] ✓ cam={cid:4s} ({cname}) "
              f"采{len(frames)}帧 检出{cam_boxes}框 "
              f"(绿{cam_counts[0]}/黄{cam_counts[1]}/红{cam_counts[2]}) "
              f"{dt}s", flush=True)

    new_labels.sort(key=lambda r: (r[0], 0 if r[4] == "-" else int(r[4].lstrip("#"))))
    with open(labels_path, "w", newline="", encoding="utf-8-sig") as lf:
        lw = csv.writer(lf)
        lw.writerow(LABEL_HEADERS)
        lw.writerows(new_labels)

    with open(status_path, "w", newline="", encoding="utf-8-sig") as sf:
        sw = csv.writer(sf)
        sw.writerow(STATUS_HEADERS)
        sw.writerows(new_status)

    print(f"\n===== 抽帧完成：在线 {online}/{len(cams)} 路，共 {total_frames} 张图、{total_boxes} 个候选框 =====")
    print(f"图片目录: {img_dir}")
    print(f"流状态表: {status_path}")
    print(f"核对考卷: {labels_path}")


def compute_metrics(rows, thresholds=None):
    """纯函数：根据 labels.csv 行列表计算分桶查准率、各阈值 PR 与按摄像头明细。"""
    if thresholds is None:
        thresholds = sorted({round(GATE, 2), 0.60, ENGINE_GATE, FLOOR}, reverse=True)
    COL_IMG = "图文件名"
    COL_CAM = "摄像头ID"
    COL_NAME = "摄像头名称"
    COL_BOX = "框编号"
    COL_CONF = "conf"
    COL_BKT = "conf分桶"
    COL_REAL = "是真人(1=是/0=假框/?=小像素忽略)"
    COL_TIGHT = "框完整(1=完整/0=截断半身)"
    COL_MISS = "未画框漏检真人数(连红黄绿框都没套住的可看清穿戴真人)"

    boxes = []
    frame_miss = defaultdict(int)
    frame_cam = {}
    cam_names = {}

    for r in rows:
        img = r[COL_IMG].strip()
        cid = r[COL_CAM].strip()
        cname = r[COL_NAME].strip()
        frame_cam[img] = cid
        cam_names[cid] = cname

        m_str = (r.get(COL_MISS) or "").strip()
        if m_str.isdigit():
            frame_miss[img] = max(frame_miss[img], int(m_str))

        if r[COL_BOX].strip() in ("-", ""):
            continue
        real_s = r[COL_REAL].strip()
        if real_s == "?":
            continue
        conf = float(r[COL_CONF])
        is_real = 1 if real_s == "1" else 0
        is_tight = 1 if (is_real and r[COL_TIGHT].strip() == "1") else 0
        boxes.append({
            "img": img,
            "cam": cid,
            "conf": conf,
            "bkt": conf_bucket(conf),
            "real": is_real,
            "tight": is_tight,
        })

    unboxed_missed = sum(frame_miss.values())
    boxed_real = sum(b["real"] for b in boxes)
    gt_total = boxed_real + unboxed_missed

    bucket_stats = {}
    for bkt in bucket_names():
        sub = [b for b in boxes if b["bkt"] == bkt]
        tot = len(sub)
        tp = sum(b["real"] for b in sub)
        tight = sum(b["tight"] for b in sub)
        fp = tot - tp
        bucket_stats[bkt] = {
            "total": tot,
            "tp": tp,
            "fp": fp,
            "tight": tight,
            "precision": (tp / tot * 100.0) if tot else None,
            "tight_rate": (tight / tot * 100.0) if tot else None,
        }

    thr_stats = []
    for thr in thresholds:
        sub = [b for b in boxes if b["conf"] >= thr]
        tp = sum(b["real"] for b in sub)
        tight = sum(b["tight"] for b in sub)
        fp = len(sub) - tp
        fn = gt_total - tp
        prec = (tp / (tp + fp) * 100.0) if (tp + fp) else None
        rec = (tp / gt_total * 100.0) if gt_total else None
        tight_rec = (tight / gt_total * 100.0) if gt_total else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec and rec and (prec + rec) > 0) else None
        thr_stats.append({
            "thr": thr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tight": tight,
            "precision": prec,
            "recall": rec,
            "tight_recall": tight_rec,
            "f1": f1,
        })

    cam_stats = {}
    for cid, cname in cam_names.items():
        c_imgs = [img for img, c in frame_cam.items() if c == cid]
        c_boxes = [b for b in boxes if b["cam"] == cid]
        ge06 = [b for b in c_boxes if b["conf"] >= GATE]
        low_real = sum(b["real"] for b in c_boxes if b["conf"] < GATE)
        miss_unboxed = sum(frame_miss[img] for img in c_imgs)
        tp06 = sum(b["real"] for b in ge06)
        fp06 = len(ge06) - tp06
        fn06 = low_real + miss_unboxed
        cam_stats[cid] = {
            "cam": cid,
            "name": cname,
            "frames": len(c_imgs),
            "tp06": tp06,
            "fp06": fp06,
            "fn06": fn06,
            "low_real": low_real,
            "miss_unboxed": miss_unboxed,
        }

    return {
        "frames": len(frame_cam),
        "boxes": len(boxes),
        "boxed_real": boxed_real,
        "unboxed_missed": unboxed_missed,
        "gt_total": gt_total,
        "buckets": bucket_stats,
        "thresholds": thr_stats,
        "cameras": cam_stats,
    }


def fmt_pct(v):
    return f"{v:.1f}%" if v is not None else "N/A"


def cmd_score(args):
    labels_path = os.path.join(args.dir, "labels.csv")
    status_path = os.path.join(args.dir, "stream_status.csv")
    report_path = os.path.join(args.dir, "report.md")
    cam_csv_path = os.path.join(args.dir, "summary_by_camera.csv")

    with open(labels_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    res = compute_metrics(rows)

    lines = [
        "# yolo26n 检人准确度评测报告（人工真值）",
        "",
        f"- 生成时间（北京时间）：`{bj_now()}`",
        f"- 核对样本：`{res['frames']}` 张全图，`{res['boxes']}` 个有效候选框（已剔除 `?` 小像素忽略框）",
        f"- 真人总数（GT）：`{res['gt_total']}` 人（其中模型 `>=0.15` 框出 `{res['boxed_real']}` 人 + 完全未画框漏检 `{res['unboxed_missed']}` 人）",
        "",
        "## 1. 各置信度分桶查准率（Precision）",
        "",
        "| 置信度分桶 | 候选框数 | 真人数 (TP) | 假框数 (FP) | 框完整真人数 | 查准率 Precision | 完整有效框率 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for bkt in bucket_names():
        b = res["buckets"][bkt]
        lines.append(
            f"| `{bkt}` | {b['total']} | {b['tp']} | {b['fp']} | {b['tight']} | "
            f"**{fmt_pct(b['precision'])}** | {fmt_pct(b['tight_rate'])} |"
        )

    lines += [
        "",
        "## 2. 不同阈值下的查准率 / 查全率对比（辅助定阈值）",
        "",
        "| 阈值 T | 检出真人 (TP) | 误框假人 (FP) | 漏检真人 (FN) | 查准率 Precision | 查全率 Recall | 完整框 Recall | F1 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for t in res["thresholds"]:
        tag = f"**{t['thr']:.2f} (现业务门)**" if abs(t["thr"] - GATE) < 1e-6 else (
              f"{t['thr']:.2f} (引擎门)" if abs(t["thr"] - ENGINE_GATE) < 1e-6 else (
              f"{t['thr']:.2f} (旧门0.6对照)" if abs(t["thr"] - 0.60) < 1e-6 and abs(GATE - 0.60) > 1e-6
              else f"{t['thr']:.2f}"))
        lines.append(
            f"| {tag} | {t['tp']} | {t['fp']} | {t['fn']} | "
            f"**{fmt_pct(t['precision'])}** | **{fmt_pct(t['recall'])}** | "
            f"{fmt_pct(t['tight_recall'])} | {fmt_pct(t['f1'])} |"
        )

    lines += [
        "",
        f"## 3. 按摄像头汇总（现业务门 >={GATE:g} 下按误检+漏检排序）",
        "",
        f"| 摄像头ID | 摄像头名称 | 抽帧数 | >={GATE:g} 真人(TP) | >={GATE:g} 假框(FP) | <{GATE:g} 被拦真人 | 未画框漏检真人 | >={GATE:g} 总漏检(FN) |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    sorted_cams = sorted(
        res["cameras"].values(),
        key=lambda x: (-(x["fp06"] + x["fn06"]), -x["fp06"], x["cam"]),
    )
    with open(cam_csv_path, "w", newline="", encoding="utf-8-sig") as cf:
        cw = csv.writer(cf)
        cw.writerow(["摄像头ID", "摄像头名称", "抽帧数", f">={GATE:g}真人(TP)", f">={GATE:g}假框(FP)",
                     f"<{GATE:g}被拦真人", "未画框漏检真人", f">={GATE:g}总漏检(FN)"])
        for c in sorted_cams:
            cw.writerow([c["cam"], c["name"], c["frames"], c["tp06"], c["fp06"],
                         c["low_real"], c["miss_unboxed"], c["fn06"]])
            lines.append(
                f"| `{c['cam']}` | {c['name']} | {c['frames']} | {c['tp06']} | "
                f"{c['fp06']} | {c['low_real']} | {c['miss_unboxed']} | {c['fn06']} |"
            )

    if os.path.exists(status_path):
        with open(status_path, encoding="utf-8-sig") as sf:
            s_rows = list(csv.DictReader(sf))
        offline = [r for r in s_rows if r.get("状态") != "在线已采"]
        lines += [
            "",
            f"## 4. RTSP 流采集覆盖率：{len(s_rows) - len(offline)}/{len(s_rows)} 路在线",
        ]
        if offline:
            lines.append("- 采集时离线/拉流失败的摄像头：" +
                         "、".join(f"`{r['摄像头ID']} ({r['摄像头名称']})`" for r in offline))

    report_txt = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write(report_txt)

    print(report_txt)
    print(f"报告已写入: {report_path}")
    print(f"分路汇总表: {cam_csv_path}")


def cmd_capture(args):
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "ERROR")
    import cv2

    cams = {c["id"]: c for c in load_cameras(args.config)}
    os.makedirs(args.out, exist_ok=True)
    ids = [x.strip() for x in args.ids.split(",") if x.strip()] or list(cams)
    for cid in ids:
        cam = cams.get(cid)
        if cam is None:
            print(f"cam={cid} 不在配置里，跳过", flush=True)
            continue
        frames, mode = pull_stream_frames(
            cv2, cam["rtsp_url"], args.per_cam, args.stride, args.per_cam * args.stride + 5
        )
        if not frames:
            print(f"cam={cid} ({cam['name']}) 拉流失败", flush=True)
            continue
        for f_i, frame in enumerate(frames, 1):
            cv2.imwrite(os.path.join(args.out, f"cam{cid}_f{f_i:02d}.jpg"), frame)
        print(f"cam={cid} ({cam['name']}) 保存 {len(frames)} 帧 ({mode})", flush=True)


def cmd_predict(args):
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "ERROR")
    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    os.makedirs(args.out, exist_ok=True)
    pred_rows = []
    times = []
    names = sorted(f for f in os.listdir(args.src) if f.endswith(".jpg"))
    model(cv2.imread(os.path.join(args.src, names[0])), imgsz=args.imgsz, conf=args.conf, classes=[0], verbose=False)
    for name in names:
        frame = cv2.imread(os.path.join(args.src, name))
        cid = name.split("_f")[0][3:]
        f_i = int(name.split("_f")[1][:2])
        t0 = time.perf_counter()
        det = model(frame, conf=args.conf, imgsz=args.imgsz, classes=[0], verbose=False)[0]
        boxes = sorted(
            ((round(float(b.conf), 2), tuple(map(int, b.xyxy[0]))) for b in det.boxes),
            key=lambda x: -x[0],
        )
        times.append((name, (time.perf_counter() - t0) * 1000))
        vis, _ = draw_annotated_frame(cv2, frame, boxes, cid, "", f_i)
        cv2.imwrite(os.path.join(args.out, name), vis)
        for b_i, (conf, (x1, y1, x2, y2)) in enumerate(boxes):
            pred_rows.append([name, cid, f"#{b_i}", f"{conf:.2f}", conf_bucket(conf), f"{x1},{y1},{x2},{y2}"])
        print(f"{name} boxes={len(boxes)}", flush=True)

    with open(os.path.join(args.out, "predictions.csv"), "w", newline="", encoding="utf-8-sig") as pf:
        pw = csv.writer(pf)
        pw.writerow(["图文件名", "摄像头ID", "框编号", "conf", "conf分桶", "bbox"])
        pw.writerows(pred_rows)
    with open(os.path.join(args.out, "timing.csv"), "w", newline="", encoding="utf-8-sig") as tf:
        tw = csv.writer(tf)
        tw.writerow(["图文件名", "推理耗时_ms"])
        tw.writerows([[n, f"{ms:.1f}"] for n, ms in times])
    avg = sum(ms for _, ms in times) / len(times)
    print(f"\n===== 预测完成：{len(names)} 张图，{len(pred_rows)} 个框（imgsz={args.imgsz}）=====")
    print(f"平均推理耗时: {avg:.1f} ms/帧")
    print(f"结果目录: {args.out}")


def cmd_selftest():
    """内置自检：验证高分假框、低分被拦真人、未画框漏检真人、?忽略框四类计算。"""
    mock_rows = [
        # 图1：1个0.8真人完整框 + 1个0.7假框 + 1个0.4低分真人框（被旧门0.6拦） + 1个未画框漏检
        {
            "图文件名": "cam1_f01.jpg", "摄像头ID": "1", "摄像头名称": "Cam1", "北京时间": "2026-11-02 15:00:00",
            "框编号": "#0", "conf": "0.80", "conf分桶": conf_bucket(0.80), "bbox": "0,0,10,10",
            "是真人(1=是/0=假框/?=小像素忽略)": "1", "框完整(1=完整/0=截断半身)": "1",
            "未画框漏检真人数(连红黄绿框都没套住的可看清穿戴真人)": "1",
        },
        {
            "图文件名": "cam1_f01.jpg", "摄像头ID": "1", "摄像头名称": "Cam1", "北京时间": "2026-11-02 15:00:00",
            "框编号": "#1", "conf": "0.70", "conf分桶": conf_bucket(0.70), "bbox": "10,10,20,20",
            "是真人(1=是/0=假框/?=小像素忽略)": "0", "框完整(1=完整/0=截断半身)": "1",
            "未画框漏检真人数(连红黄绿框都没套住的可看清穿戴真人)": "",
        },
        {
            "图文件名": "cam1_f01.jpg", "摄像头ID": "1", "摄像头名称": "Cam1", "北京时间": "2026-11-02 15:00:00",
            "框编号": "#2", "conf": "0.40", "conf分桶": conf_bucket(0.40), "bbox": "20,20,30,30",
            "是真人(1=是/0=假框/?=小像素忽略)": "1", "框完整(1=完整/0=截断半身)": "0",
            "未画框漏检真人数(连红黄绿框都没套住的可看清穿戴真人)": "",
        },
        {
            "图文件名": "cam1_f01.jpg", "摄像头ID": "1", "摄像头名称": "Cam1", "北京时间": "2026-11-02 15:00:00",
            "框编号": "#3", "conf": "0.20", "conf分桶": conf_bucket(0.20), "bbox": "30,30,40,40",
            "是真人(1=是/0=假框/?=小像素忽略)": "?", "框完整(1=完整/0=截断半身)": "1",
            "未画框漏检真人数(连红黄绿框都没套住的可看清穿戴真人)": "",
        },
        # 图2：无检出框，但有2个未画框真人
        {
            "图文件名": "cam2_f01.jpg", "摄像头ID": "2", "摄像头名称": "Cam2", "北京时间": "2026-11-02 15:00:00",
            "框编号": "-", "conf": "-", "conf分桶": BUCKET_NONE, "bbox": "-",
            "是真人(1=是/0=假框/?=小像素忽略)": "-", "框完整(1=完整/0=截断半身)": "-",
            "未画框漏检真人数(连红黄绿框都没套住的可看清穿戴真人)": "2",
        },
    ]
    m = compute_metrics(mock_rows, thresholds=(0.60, 0.25))
    assert m["gt_total"] == 5, f"expect gt_total=5, got {m['gt_total']}"
    t06 = m["thresholds"][0]
    assert t06["tp"] == 1 and t06["fp"] == 1 and t06["fn"] == 4
    assert abs(t06["precision"] - 50.0) < 1e-6
    assert abs(t06["recall"] - 20.0) < 1e-6
    t025 = m["thresholds"][1]
    assert t025["tp"] == 2 and t025["fp"] == 1 and t025["fn"] == 3
    assert abs(t025["tight_recall"] - 20.0) < 1e-6
    print("selftest passed (5 assertions ok)")


def main():
    global GATE
    ap = argparse.ArgumentParser(description="yolo26n 检人准确度沙箱抽样与算分工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_sample = sub.add_parser("sample", help="逐路拉取全部 RTSP 流，跑 yolo26n 预画框并生成 labels.csv")
    sp_sample.add_argument("--config", default="config.yaml", help="配置文件路径")
    sp_sample.add_argument("--model", default="/tmp/p.pt", help="yolo26n 权重路径")
    sp_sample.add_argument("--out", default="/tmp/sandbox", help="沙箱输出目录")
    sp_sample.add_argument("--per-cam", type=int, default=3, help="每路摄像头抽取帧数")
    sp_sample.add_argument("--stride", type=int, default=12, help="抽帧间隔帧数（约1秒）")
    sp_sample.add_argument("--conf", type=float, default=0.15, help="检测置信度下限")
    sp_sample.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸（默认640，对比用1280）")
    sp_sample.add_argument("--gate", type=float, default=0.30, help="业务门 person_conf_threshold（绿/黄分界）")
    sp_sample.add_argument("--only", default="", help="仅重采指定摄像头ID（逗号分隔，保留其余路数据）")
    sp_sample.add_argument("--retry-offline", action="store_true", help="仅重采上次离线失败的摄像头（保留已采路数据）")

    sp_score = sub.add_parser("score", help="读取人工核对后的 labels.csv 计算查准率/查全率")
    sp_score.add_argument("--dir", default="sandbox", help="含 labels.csv 的沙箱目录")
    sp_score.add_argument("--gate", type=float, default=0.30, help="业务门（须与 sample 时一致）")

    sp_cap = sub.add_parser("capture", help="按指定摄像头实时拉流，保存原始帧（无标注）")
    sp_cap.add_argument("--config", default="config.yaml", help="配置文件路径")
    sp_cap.add_argument("--ids", default="", help="摄像头ID列表（逗号分隔，留空=config 中全部启用路）")
    sp_cap.add_argument("--out", default="sandbox/raw", help="原始帧输出目录")
    sp_cap.add_argument("--per-cam", type=int, default=3, help="每路抽取帧数")
    sp_cap.add_argument("--stride", type=int, default=12, help="抽帧间隔帧数")

    sp_pred = sub.add_parser("predict", help="对已有图片目录跑推理并存结果（不需要 RTSP）")
    sp_pred.add_argument("--src", default="sandbox/images", help="输入图片目录")
    sp_pred.add_argument("--out", default="sandbox/image-1280", help="结果输出目录")
    sp_pred.add_argument("--model", default="/tmp/p.pt", help="yolo26n 权重路径")
    sp_pred.add_argument("--imgsz", type=int, default=1280, help="推理输入尺寸")
    sp_pred.add_argument("--conf", type=float, default=0.15, help="检测置信度下限")
    sp_pred.add_argument("--gate", type=float, default=0.30, help="业务门（绿/黄分界，用于画框颜色）")

    sub.add_parser("selftest", help="运行算分逻辑单元测试")

    args = ap.parse_args()
    if args.cmd == "sample":
        GATE = args.gate
        cmd_sample(args)
    elif args.cmd == "capture":
        cmd_capture(args)
    elif args.cmd == "predict":
        GATE = args.gate
        cmd_predict(args)
    elif args.cmd == "score":
        GATE = args.gate
        cmd_score(args)
    elif args.cmd == "selftest":
        cmd_selftest()


if __name__ == "__main__":
    main()
