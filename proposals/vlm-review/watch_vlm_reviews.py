#!/usr/bin/env python3
"""
抓取 VLM 二级复核结论（只读 tail 生产日志，零侵入，不影响检测进程）

从 deploy/logs/server.log 里挑出 `vlm-review` 行（含 verdict/frames/耗时），
逐条打印并在退出时汇总统计——shadow 灰度期"核对一周 verdict 对错 + total 分布"的数据源。
仿 tools/watch_alerts.py 的外部观察模式。

用法:
    python3 proposals/vlm-review/watch_vlm_reviews.py            # 跟随当前日志（Ctrl-C 出汇总）
    python3 proposals/vlm-review/watch_vlm_reviews.py --once deploy/logs/server*
        # 扫全部历史轮转包（.gz 自动解压）+当前日志，出整周汇总——灰度周复盘用这条
    加 --csv 路径 → 每条判定追加写 CSV（行级去重、幂等，可反复跑/反复重启；Excel 直接打开）。
    常驻自动收集（机器重启后需再拉起；pgrep -f watch_vlm_reviews 查活体）：
        nohup python3 proposals/vlm-review/watch_vlm_reviews.py \
            --csv proposals/vlm-review/vlm-review-result.csv >/dev/null 2>&1 &

判定行格式（写侧见 server/alert/manager.py::_vlm_review）:
    vlm-review camera=.. rule=.. verdict=yes|no|skipped frames=N sample=Xms infer=Yms total=Zms
    时间按**北京时间**显示（日志原生 UTC，+8h）；行首时刻 ≈ 该告警 🚨 触发时刻，
    shadow 模式下同秒告警均已推送 → 下游/payload 图即本条复核所看的证据帧。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def _read_lines(path: Path):
    """普通/gz 日志逐行读出（二进制行，交给调用方 parse）。"""
    fp = gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")
    with fp:
        yield from fp

LINE_RE = re.compile(
    r"vlm-review camera=(?P<camera>.*?) rule=(?P<rule>.*?) "
    r"verdict=(?P<verdict>\w+) frames=(?P<frames>\d+) "
    r"sample=(?P<sample>\d+)ms infer=(?P<infer>\d+)ms total=(?P<total>\d+)ms"
)

EXPECT = {}   # 两问合一后三规则极性统一 yes=违规坐实（与 vlm_review.REVIEWS 同步）


def mark(rule: str, verdict: str) -> str:
    if verdict == "skipped":
        return "∅跳"
    return "✓确认" if verdict == EXPECT.get(rule, "yes") else "✗排除"


def fmt(camera: str, rule: str, verdict: str, frames: str, sample: str,
        infer: str, total: str, ts: str) -> str:
    return f"[{ts}] {mark(rule, verdict)} camera={camera} rule={rule} " \
           f"verdict={verdict} frames={frames} " \
           f"sample={sample}ms infer={infer}ms total={total}ms"


class Stats:
    def __init__(self):
        self.outcomes = Counter()   # 按"违规确认/排除/跳过"计，极性无关
        self.by_rule = {}           # rule -> Counter(outcome)
        self.totals: list[int] = []
        self.infers: list[int] = []

    def add(self, m: re.Match):
        outcome = mark(m["rule"], m["verdict"])
        self.outcomes[outcome] += 1
        self.by_rule.setdefault(m["rule"], Counter())[outcome] += 1
        self.totals.append(int(m["total"]))
        self.infers.append(int(m["infer"]))

    def summary(self) -> str:
        if not self.totals:
            return "（无 vlm-review 记录）"
        t = sorted(self.totals)
        lines = [f"共 {sum(self.outcomes.values())} 条: " +
                 ", ".join(f"{k}={v}" for k, v in self.outcomes.items()) +
                 f" | total ms 中位={t[len(t)//2]} 最大={t[-1]}" +
                 f" | infer 中位={sorted(self.infers)[len(self.infers)//2]}"]
        for rule, c in self.by_rule.items():   # 分规则看确认率=拦对没拦对
            lines.append(f"  {rule}: " + ", ".join(f"{k}{v}" for k, v in c.items()))
        return "\n".join(lines)


def iter_new_lines(path: Path, follow: bool):
    """按字节偏移产出日志行。--once 读完现有内容即止；follow 持续追新增，
    未写完的半行留待下次（偏移只推进到最后一个完整行尾）。"""
    pos = 0
    while True:
        with path.open("rb") as fp:
            fp.seek(pos)
            chunk = fp.read()
        if chunk:
            end = chunk.rfind(b"\n") + 1        # 最后一个完整行（含\n）的结束位置
            if end:
                pos += end
                for raw in chunk[:end].split(b"\n")[:-1]:
                    yield raw
            elif not follow:
                return
        else:
            if not follow:
                return
            if path.stat().st_size < pos:       # 轮转/清空 → 从头读
                pos = 0
        if follow:
            time.sleep(1.0)


def make_csv(path: str):
    """返回 emit(row) 闭包：追加写 + 行级去重（重跑/重启幂等）；新文件带表头+BOM。"""
    p = Path(path)
    fresh = not p.exists() or p.stat().st_size == 0
    seen = set()
    if not fresh:
        with p.open(encoding="utf-8-sig", newline="") as fp:
            seen = {tuple(r) for r in csv.reader(fp)}
    fp = p.open("a", newline="", encoding="utf-8-sig" if fresh else "utf-8")
    w = csv.writer(fp)
    if fresh:
        w.writerow(("北京时间", "camera", "rule", "verdict", "结果",
                    "frames", "sample_ms", "infer_ms", "total_ms"))
        fp.flush()

    def emit(row):
        if tuple(row) in seen:      # 完全重复的行（重启回填时）跳过
            return
        seen.add(tuple(row))
        w.writerow(row)
        fp.flush()
    return emit


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取 vlm-review 复核日志")
    ap.add_argument("paths", nargs="*", metavar="LOG",
                    default=["deploy/logs/server.log"],
                    help="一个或多个日志文件（.gz 自动解压；shell 通配符展开传入）。跟随模式只用第一个")
    ap.add_argument("--once", action="store_true", help="扫完给定文件出汇总即退出")
    ap.add_argument("--csv", default=None, help="判定实时追加写入的 CSV 路径（去重幂等）")
    args = ap.parse_args()

    stats = Stats()
    emit = make_csv(args.csv) if args.csv else None

    def handle(raw: bytes):
        try:
            text = json.loads(raw.decode("utf-8", errors="replace"))["text"]
        except Exception:
            return
        m = LINE_RE.search(text)
        if m is None:
            return
        stats.add(m)
        # 日志时间戳是 UTC(+0000)；显示北京时间(+8h，含日期——跨天样本别对错)。
        # 对应证据帧：同摄像头、同秒的 🚨 alert 即本条复核对象（shadow 下都推送了）。
        try:
            dt = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S") + timedelta(hours=8)
            ts = dt.strftime("%m-%d %H:%M:%S")
        except Exception:
            ts = ""
        print(fmt(m["camera"], m["rule"], m["verdict"], m["frames"],
                  m["sample"], m["infer"], m["total"], ts), flush=True)
        if emit:
            emit((ts, m["camera"], m["rule"], m["verdict"],
                  mark(m["rule"], m["verdict"]), m["frames"],
                  m["sample"], m["infer"], m["total"]))

    if args.once:
        for pth in args.paths:
            p = Path(pth)
            if not p.exists():
                print(f"跳过（不存在）: {p}", file=sys.stderr)
                continue
            for raw in _read_lines(p):
                handle(raw)
    else:
        path = Path(args.paths[0])
        if not path.exists():
            print(f"日志不存在: {path}", file=sys.stderr)
            return 2
        for raw in iter_new_lines(path, follow=True):
            handle(raw)

    print("\n===== 汇总 =====", flush=True)
    print(stats.summary(), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已停止", flush=True)
        raise SystemExit(0)
