#!/usr/bin/env python3
"""smoking 时序终审 · 生产实时监测（只读，宿主机运行，Ctrl-C 退出）。

盯 `docker logs safety-detector`，把吸烟链路逐事件打出来并落 CSV：
  触发(alert) → 复核(vlm-review 8帧时序, score=small/vlm/avg) → 推送 or 丢弃

    python3 temp/live-smoking-win/monitor_prod_smoking.py            # 默认从最近10分钟起
    python3 temp/live-smoking-win/monitor_prod_smoking.py --since 1h

输出：控制台表格 + 追加 prod-smoke-<日期>.csv（同目录）。
frames 列应为 8 —— 出现 4 说明跑的还是旧版复核。
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
VLM_RE = re.compile(r"vlm-review camera=(?P<cam>.+?) rule=smoking verdict=(?P<verdict>\w+) "
                    r"frames=(?P<frames>\d+) sample=(?P<sample>\d+)ms infer=(?P<infer>\d+)ms "
                    r"total=(?P<total>\d+)ms.*?score=small=(?P<small>[\d.]+) "
                    r"vlm=(?P<vlm>[\d.]+) avg=(?P<avg>[\d.]+)")
VLM_NOSCORE_RE = re.compile(r"vlm-review camera=(?P<cam>.+?) rule=smoking verdict=(?P<verdict>\w+) "
                            r"frames=(?P<frames>\d+)")
DROP_RE = re.compile(r"vlm-review camera=(?P<cam>.+?) rule=smoking "
                     r"score=small=(?P<small>[\d.]+) vlm=(?P<vlm>[\d.]+) "
                     r"avg=(?P<avg>[\d.]+) → 丢弃\(阈值 (?P<thr>[\d.]+)\)")
ALERT_RE = re.compile(r"alert camera=(?P<cam>.+?) rule=smoking class=person conf=(?P<conf>[\d.]+)")
PUSH_RE = re.compile(r"Webhook 推送成功 .*alert_type=smoking")
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")

HEAD = ["时间(北京)", "UTC时间", "摄像头", "事件", "verdict", "frames",
        "小模型", "大模型时序", "平均分", "sample_ms", "infer_ms", "total_ms", "备注"]


def bj(ts_utc):
    t = datetime.strptime(ts_utc, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
    return (t + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="10m", help="docker logs --since，如 10m/1h")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()
    csv_path = os.path.join(a.out, f"prod-smoke-{datetime.now():%Y%m%d}.csv")
    new = not os.path.exists(csv_path)
    fout = open(csv_path, "a", newline="", encoding="utf-8-sig")
    w = csv.writer(fout)
    if new:
        w.writerow(HEAD)
    print(f"监测 safety-detector smoking 时序终审 → {csv_path}  (Ctrl-C 退出)", flush=True)

    def emit(ts, cam, event, **f):
        row = [bj(ts) if ts else "-", ts or "-", cam, event,
               f.get("verdict", "-"), f.get("frames", "-"), f.get("small", "-"),
               f.get("vlm", "-"), f.get("avg", "-"), f.get("sample", "-"),
               f.get("infer", "-"), f.get("total", "-"), f.get("note", "")]
        w.writerow(row); fout.flush()
        print(" | ".join(f"{HEAD[i]}:{v}" for i, v in enumerate(row)
                         if v not in ("-", "")), flush=True)

    while True:
        try:
            p = subprocess.Popen(["docker", "logs", "-f", "--since", a.since, "safety-detector"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        except FileNotFoundError:
            sys.exit("找不到 docker 命令")
        for line in p.stdout:
            line = ANSI.sub("", line.rstrip())
            m = TS_RE.match(line)
            ts = m.group(1) if m else ""
            if "rule=smoking" not in line and "alert_type=smoking" not in line:
                continue
            m = ALERT_RE.search(line)
            if m:
                emit(ts, m["cam"], "触发", note=f"person conf={m['conf']}")
                continue
            m = DROP_RE.search(line)
            if m:
                g = m.groupdict()
                emit(ts, g["cam"], "丢弃(门控)", small=g["small"], vlm=g["vlm"], avg=g["avg"],
                     note=f"avg≤阈值{g['thr']}")
                continue
            m = VLM_RE.search(line)
            if m:
                g = m.groupdict()
                warn = "⚠ 旧版4帧" if g["frames"] == "4" else (
                    "⚠ verdict=no 但 avg 过线" if float(g["avg"]) > 0.75 and g["verdict"] == "no"
                    else "")
                emit(ts, g["cam"], "复核", verdict=g["verdict"], frames=g["frames"],
                     small=g["small"], vlm=g["vlm"], avg=g["avg"], sample=g["sample"],
                     infer=g["infer"], total=g["total"], note=warn)
                continue
            m = VLM_NOSCORE_RE.search(line)
            if m and "丢弃" not in line:        # verdict=skipped 等拿不到分数的复核行
                emit(ts, m["cam"], "复核(无分数)", verdict=m["verdict"], frames=m["frames"],
                     note="vlm不可用→fail_open" )
                continue
            if "rule=smoking" in line and "丢弃(fail_open=false)" in line:
                m2 = re.search(r"camera=(.+?) rule=smoking", line)
                emit(ts, m2.group(1) if m2 else "-", "丢弃(vlm不可用)")
                continue
            if PUSH_RE.search(line):
                emit(ts, "-", "推送(webhook已收)")
        time.sleep(1)          # 日志流断开（容器重启）→ 自动重连
        print("[monitor] 日志流断开，1s 后重连", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出", flush=True)
