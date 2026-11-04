#!/usr/bin/env python3
"""从生产容器日志抓 smoking 复核的**真实分数**（小模型/vlm/平均），写表。

比“从平台证据图离线重算”准确：
  - 小模型分 = 触发帧烟头属性的 conf（生产值，不是存档图重算）
  - 大模型分 = 生产 4 帧连拍宫格的 P(yes)
  - 被门控丢弃的事件也会记录（平台里看不到，因为它根本没推送）

宿主机运行（仅标准库，需要 docker CLI）：
    python3 sandbox/watch_prod_scores.py --cams 2639,1924 --duration 7200

输出：temp/smoking-confidence-2639-1924-prod.{csv,md}
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

ANSI = re.compile(r"\x1b\[[0-9;]*m")
EV_RE = re.compile(r"evidence camera=(?P<cam>\S+) rule=(?P<rule>\w+)")
ALERT_RE = re.compile(r"alert camera=(?P<name>.+?) rule=(?P<rule>\w+) class=")
VLM_RE = re.compile(
    r"vlm-review camera=(?P<name>.+?) rule=(?P<rule>\w+) verdict=(?P<verdict>\w+) "
    r"frames=(?P<frames>\d+) sample=(?P<sample>\d+)ms infer=(?P<infer>\d+)ms "
    r"total=(?P<total>\d+)ms persons=(?P<persons>\d+) score=(?P<score>.+)$")
DROP_RE = re.compile(r"vlm-review camera=.* rule=(?P<rule>\w+) score=.* → 丢弃")
ROW_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?(?P<msg>(?:evidence|vlm-review).*)$")

HEAD = ["时间(北京)", "摄像头ID", "规则", "verdict", "帧数", "小模型conf", "大模型P(yes)",
        "平均分", "生产门控", "sample_ms", "infer_ms", "total_ms"]


def bj(ts: str) -> str:
    """容器日志是 UTC，转北京时间。"""
    return (datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc) + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def write(path_csv, path_md, rows):
    with open(path_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(HEAD); w.writerows(rows)
    lines = ["# 2639/1924 smoking 生产真实分数（小模型 + 大模型 4 帧宫格）", "",
             "> 数据源：`docker logs safety-detector` 的 `vlm-review ... score=small/vlm/avg`。",
             "> **包含被门控丢弃的事件**（平台里看不到）。时间为北京时间。", "",
             "| " + " | ".join(HEAD) + " |",
             "|" + "|".join([":---"] * len(HEAD)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    if not rows:
        lines += ["", "_暂无 smoking 事件_"]
    open(path_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cams", default="2639,1924")
    ap.add_argument("--container", default="safety-detector")
    ap.add_argument("--duration", type=int, default=7200, help="运行时长秒")
    ap.add_argument("--out", default="temp")
    a = ap.parse_args()
    cams = {c.strip() for c in a.cams.split(",") if c.strip()}
    os.makedirs(a.out, exist_ok=True)
    tag = "-".join(c for c in a.cams.split(",") if c.strip())   # 保留 --cams 的书写顺序
    path_csv = os.path.join(a.out, f"smoking-confidence-{tag}-prod.csv")
    path_md = os.path.join(a.out, f"smoking-confidence-{tag}-prod.md")

    bj_now = datetime.now(timezone.utc) + timedelta(hours=8)
    since = (bj_now.replace(hour=0, minute=0, second=0) - timedelta(hours=8))
    cmd = ["docker", "logs", "-f", "--since", since.strftime("%Y-%m-%dT%H:%M:%S"),
           a.container]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    rows, seen = [], set()
    last_evidence = {}          # rule -> camera id（vlm-review 行只有名字，靠前一条 evidence 关联）
    pending = None              # (cam, ts, monotonic)：等待同线程紧随的“丢弃”日志
    t_end = time.monotonic() + a.duration
    write(path_csv, path_md, rows)
    print(f"watch prod logs cams={cams} duration={a.duration}s", flush=True)

    def resolve_pending(decision: str):
        """把最后一条“待定”行落定为 推送/丢弃。"""
        for row in reversed(rows):
            if row[8] == "待定":
                row[8] = decision
                write(path_csv, path_md, rows)
                return

    try:
        for line in proc.stdout:
            if time.monotonic() > t_end:
                break
            line = ANSI.sub("", line.rstrip("\n"))
            # 超过 3 秒没有“丢弃”日志 → 该事件已推送（丢弃日志与复核行同线程、紧随其后）
            if pending and time.monotonic() - pending[2] > 3.0:
                resolve_pending("推送")
                pending = None
            m = ROW_RE.match(line)
            if not m:
                continue
            ts, msg = m.group("ts"), m.group("msg")
            ev = EV_RE.search(msg)
            if ev:
                last_evidence[ev.group("rule")] = ev.group("cam")
                continue
            mv = VLM_RE.search(msg)
            if mv and mv.group("rule") == "smoking":
                cam = last_evidence.get("smoking", "")
                if cam not in cams:
                    continue
                score = mv.group("score")
                mm = re.search(r"small=([\d.]+) vlm=([\d.]+) avg=([\d.]+)", score)
                if not mm:
                    continue
                key = (cam, ts, mm.group(3))
                if key in seen:
                    continue
                seen.add(key)
                row = [bj(ts), cam, "smoking", mv.group("verdict"), mv.group("frames"),
                       mm.group(1), mm.group(2), mm.group(3), "待定",
                       mv.group("sample"), mv.group("infer"), mv.group("total")]
                rows.append(row)
                pending = (cam, ts, time.monotonic())
                write(path_csv, path_md, rows)
                print("NEW", row[:9], flush=True)
                continue
            md = DROP_RE.search(msg)
            if md and md.group("rule") == "smoking" and pending:
                resolve_pending("丢弃")
                pending = None
                continue
            ma = ALERT_RE.search(msg)
            if ma and ma.group("rule") == "smoking" and pending:
                # 新告警到来前没有丢弃日志 → 上一条已推送
                resolve_pending("推送")
                pending = None
    finally:
        proc.terminate()
        resolve_pending("推送")
    print(f"done rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
