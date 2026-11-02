#!/usr/bin/env python3
"""从 temp2/vlm-audit 构建固定考卷：按事件抽样、尺寸/相机分层、补负样本。

输出目录：temp/vlm-audit-exam/
  images/          200 个事件 × 4 张图
  exam.csv         一行一个事件，人工标注列留空
  negatives/       负样本图（已戴帽 / 已穿衣）
  negatives.csv    一行一个负样本，人工标注列留空
  README.md        标注说明与抽样口径
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path.cwd()  # 在仓库根目录运行；若从本目录运行则回退到仓库根
if not (ROOT / "proposals").is_dir():
    ROOT = Path(__file__).resolve().parents[2]
SRC_CSV = ROOT / "proposals/vlm-review/vlm-audit-result.csv"
SRC_IMG = ROOT / "temp2/vlm-audit"
OUT = ROOT / "temp/vlm-audit-exam"
EXAM_N = 200
NEG_PER_CLASS = 50
SEED = 20261102
BUCKETS = ["<32", "32~64", "64~128", ">128"]
PARTS = ("person", "head", "torso", "frame")


def bucket_of(w: int) -> str:
    if w < 32:
        return "<32"
    if w < 64:
        return "32~64"
    if w < 128:
        return "64~128"
    return ">128"


def alloc(total: int, weights: dict) -> dict:
    """最大余数法：把 total 按权重分给各 key，且不超过权重（权重即容量）。"""
    if total <= 0 or not weights:
        return {k: 0 for k in weights}
    if total >= sum(weights.values()):
        return dict(weights)
    s = sum(weights.values())
    raw = {k: total * v / s for k, v in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    left = total - sum(out.values())
    order = sorted(weights, key=lambda k: (raw[k] - out[k], weights[k]), reverse=True)
    for k in order[:left]:
        out[k] += 1
    return out


def sample_stratified(ids, n, events, rng):
    """按相机分层抽 n 个事件；相机数允许时每路至少 1 个。"""
    if n >= len(ids):
        return sorted(ids)
    groups = defaultdict(list)
    for eid in ids:
        groups[events[eid]["camera"]].append(eid)
    cams = sorted(groups)
    if n >= len(cams):
        quota = {c: 1 for c in cams}
        caps = {c: len(groups[c]) - 1 for c in cams}
        extra = alloc(n - len(cams), caps)
        for c in cams:
            quota[c] += extra.get(c, 0)
    else:
        quota = alloc(n, {c: len(groups[c]) for c in cams})

    out = []
    for c in cams:
        q = min(quota.get(c, 0), len(groups[c]))
        out.extend(rng.sample(sorted(groups[c]), q))
    if len(out) < n:  # 取整/容量兜底
        rest = sorted(set(ids) - set(out))
        out.extend(rng.sample(rest, n - len(out)))
    return sorted(out)


def main():
    exam_csv = OUT / "exam.csv"
    if exam_csv.exists():
        old = list(csv.DictReader(exam_csv.open(encoding="utf-8-sig", newline="")))
        if any(r.get("人工_安全帽") or r.get("人工_反光衣") for r in old):
            raise SystemExit("exam.csv 已有人工标注，拒绝覆盖；请先备份或移走本目录")

    rows = list(csv.DictReader(SRC_CSV.open(encoding="utf-8-sig", newline="")))
    byid = defaultdict(list)
    for r in rows:
        byid[Path(r["图前缀"]).name].append(r)

    events = {}
    for eid, rs in byid.items():
        paths = {p: SRC_IMG / f"{eid}_{p}.jpg" for p in PARTS}
        if not all(p.is_file() and p.stat().st_size > 1024 for p in paths.values()):
            continue
        boxes = {(int(r["bbox_x1"]), int(r["bbox_y1"]),
                  int(r["bbox_x2"]), int(r["bbox_y2"])) for r in rs}
        x1, y1, x2, y2 = sorted(boxes)[0]
        rules = sorted({r["rule"] for r in rs})
        events[eid] = {
            "eid": eid, "time": rs[0]["北京时间"], "camera": rs[0]["camera"],
            "x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1,
            "bucket": bucket_of(x2 - x1), "rules": rules, "paths": paths,
            # 单规则事件里另一项没有被判违规（yes 或请求失败，需人工确认）
            "model_helmet": "no" if "no_helmet" in rules else "yes(未违规)",
            "model_vest": "no" if "no_vest" in rules else "yes(未违规)",
        }

    by_bucket = defaultdict(list)
    for eid, ev in events.items():
        by_bucket[ev["bucket"]].append(eid)

    # 尺寸分层：稀有桶全取，其余名额按事件数比例分给中间桶。
    rare = {b for b in BUCKETS if len(by_bucket[b]) <= 40}
    picked = {b: len(by_bucket[b]) if b in rare else 0 for b in BUCKETS}
    left = EXAM_N - sum(picked.values())
    if left > 0:
        weights = {b: len(by_bucket[b]) for b in BUCKETS if b not in rare}
        for b, n in alloc(left, weights).items():
            picked[b] = n
    elif left < 0:
        picked = alloc(EXAM_N, {b: len(by_bucket[b]) for b in BUCKETS})

    rng = random.Random(SEED)
    exam = []
    for b in BUCKETS:
        exam.extend(sample_stratified(by_bucket[b], picked[b], events, rng))
    exam = sorted(exam, key=lambda e: (BUCKETS.index(events[e]["bucket"]),
                                       events[e]["camera"], e))
    assert len(exam) == EXAM_N, f"expected {EXAM_N}, got {len(exam)}"

    exam_set = set(exam)
    neg_helmet = [e for e, ev in events.items()
                  if ev["rules"] == ["no_vest"] and e not in exam_set]
    neg_vest = [e for e, ev in events.items()
                if ev["rules"] == ["no_helmet"] and e not in exam_set]
    neg_helmet = sample_stratified(neg_helmet, min(NEG_PER_CLASS, len(neg_helmet)),
                                   events, rng)
    neg_vest = sample_stratified(neg_vest, min(NEG_PER_CLASS, len(neg_vest)),
                                 events, rng)

    out_img = OUT / "images"
    out_neg = OUT / "negatives"
    for d in (out_img, out_neg):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for eid in exam:
        for part, src in events[eid]["paths"].items():
            shutil.copy2(src, out_img / f"{eid}_{part}.jpg")

    neg_rows = []
    for eid in neg_helmet:
        ev = events[eid]
        src = ev["paths"]["head"]
        dst = out_neg / f"{eid}_head.jpg"
        shutil.copy2(src, dst)
        neg_rows.append({
            "事件ID": eid, "时间": ev["time"], "相机": ev["camera"],
            "类别": "安全帽-已戴", "来源": "巡检单规则事件",
            "模型规则": "|".join(ev["rules"]), "bbox_w": ev["w"], "bbox_h": ev["h"],
            "尺寸桶": ev["bucket"], "图片": f"negatives/{dst.name}",
            "模型_安全帽": "yes(未违规)", "模型_反光衣": "-",
            "人工_安全帽": "", "人工_反光衣": "-",
        })
    for eid in neg_vest:
        ev = events[eid]
        src = ev["paths"]["torso"]
        dst = out_neg / f"{eid}_torso.jpg"
        shutil.copy2(src, dst)
        neg_rows.append({
            "事件ID": eid, "时间": ev["time"], "相机": ev["camera"],
            "类别": "反光衣-已穿", "来源": "巡检单规则事件",
            "模型规则": "|".join(ev["rules"]), "bbox_w": ev["w"], "bbox_h": ev["h"],
            "尺寸桶": ev["bucket"], "图片": f"negatives/{dst.name}",
            "模型_安全帽": "-", "模型_反光衣": "yes(未违规)",
            "人工_安全帽": "-", "人工_反光衣": "",
        })

    exam_cols = [
        "事件ID", "时间", "相机", "模型规则", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "bbox_w", "bbox_h", "尺寸桶", "模型_安全帽", "模型_反光衣",
        "图片_person", "图片_head", "图片_torso", "图片_frame",
        "人工_安全帽", "人工_反光衣",
    ]
    with (OUT / "exam.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=exam_cols)
        w.writeheader()
        for eid in exam:
            ev = events[eid]
            w.writerow({
                "事件ID": eid, "时间": ev["time"], "相机": ev["camera"],
                "模型规则": "|".join(ev["rules"]),
                "bbox_x1": ev["x1"], "bbox_y1": ev["y1"],
                "bbox_x2": ev["x2"], "bbox_y2": ev["y2"],
                "bbox_w": ev["w"], "bbox_h": ev["h"], "尺寸桶": ev["bucket"],
                "模型_安全帽": ev["model_helmet"], "模型_反光衣": ev["model_vest"],
                "图片_person": f"images/{eid}_person.jpg",
                "图片_head": f"images/{eid}_head.jpg",
                "图片_torso": f"images/{eid}_torso.jpg",
                "图片_frame": f"images/{eid}_frame.jpg",
                "人工_安全帽": "", "人工_反光衣": "",
            })

    neg_cols = [
        "事件ID", "时间", "相机", "类别", "来源", "模型规则", "bbox_w", "bbox_h",
        "尺寸桶", "图片", "模型_安全帽", "模型_反光衣", "人工_安全帽", "人工_反光衣",
    ]
    with (OUT / "negatives.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=neg_cols)
        w.writeheader()
        w.writerows(sorted(neg_rows, key=lambda r: (r["类别"], r["相机"], r["事件ID"])))

    print(f"exam events: {len(exam)}")
    print("bucket counts:", {b: sum(1 for e in exam if events[e]["bucket"] == b)
                             for b in BUCKETS})
    print("cameras:", len({events[e]["camera"] for e in exam}))
    print(f"negative helmet-wearing: {len(neg_helmet)}")
    print(f"negative vest-wearing: {len(neg_vest)}")
    print(f"images: {sum(1 for _ in out_img.iterdir())} + negatives: "
          f"{sum(1 for _ in out_neg.iterdir())}")


if __name__ == "__main__":
    main()
