#!/usr/bin/env python3
"""
告警监控（只读 tail 生产日志，不干扰检测进程）

用途: 盯住指定摄像头的告警，出现即打印/落文件，可选推送到 webhook。
它不改配置、不连检测进程、不重新拉流，只读 deploy/logs/server.log。

为什么不在检测进程里加钩子: 生产正跑着，任何代码改动都要重启 24 路管线；
这个脚本从外部观察，零侵入。

用法:
    python3 tools/watch_alerts.py --cameras "AI Camera0670,AI Camera0880" \
        --out /tmp/alerts-670-880.log

    # 推到企业微信/钉钉机器人
    python3 tools/watch_alerts.py --cameras "AI Camera0670,AI Camera0880" \
        --webhook 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx'

    # 只跑 2 小时
    python3 tools/watch_alerts.py --duration 7200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path


def _notify(url: str, text: str, timeout: float = 5.0) -> None:
    """尽力推送；失败只告警不中断监控。兼容企业微信/钉钉的 text 消息体。"""
    body = json.dumps({"msgtype": "text", "text": {"content": text}}).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
    except Exception as exc:
        print(f"  [推送失败] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)


def _read_from(path: Path, pos: int) -> tuple[bytes, int]:
    """从字节偏移 pos 读新增内容。返回 (数据, 新偏移)。

    用二进制 + 显式偏移，而不是长期持有文本句柄：文本模式在 EOF 之后不会自动
    重新读取追加内容，会漏掉所有新告警。
    """
    with path.open("rb") as fp:
        fp.seek(pos)
        chunk = fp.read()
    return chunk, pos + len(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(description="指定摄像头的告警监控")
    ap.add_argument("--log", default="deploy/logs/server.log")
    ap.add_argument("--cameras", default="AI Camera0670,AI Camera0880",
                    help="逗号分隔的摄像头名称（日志里的 camera= 值）")
    ap.add_argument("--out", default=None, help="同时追加写入该文件")
    ap.add_argument("--webhook", default=None, help="告警推送到该 URL（企业微信/钉钉机器人）")
    ap.add_argument("--duration", type=float, default=0, help="运行多少秒后退出，0=一直跑")
    ap.add_argument("--interval", type=float, default=1.0, help="无新行时的轮询间隔")
    args = ap.parse_args()

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    path = Path(args.log)
    if not path.exists():
        print(f"日志不存在: {path}", file=sys.stderr)
        return 2

    st = path.stat()
    inode, pos, buf = st.st_ino, st.st_size, b""   # 从当前末尾开始，只看新告警
    started = time.time()
    print(f"监控中: {cams}  ← {path}", flush=True)
    print("（只读日志，不影响检测；Ctrl-C 退出）", flush=True)

    try:
        while True:
            try:
                st = path.stat()
            except OSError:
                st = None

            # 日志轮转（inode 变）或被截断（size 变小）→ 从头读新文件
            if st is None or st.st_ino != inode or st.st_size < pos:
                if st is not None and st.st_ino != inode:
                    print("  (日志已轮转，重新打开)", flush=True)
                inode = st.st_ino if st else inode
                pos, buf = 0, b""

            if st is not None and st.st_size > pos:
                chunk, pos = _read_from(path, pos)
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8", errors="replace")
                    if "🚨 alert" not in line:
                        continue
                    try:
                        rec = json.loads(line)
                        text, ts = rec["text"], rec["record"]["time"]["repr"][:19]
                    except Exception:
                        continue
                    if not any(c in text for c in cams):
                        continue
                    msg = f"[{ts}] {text.strip()}"
                    print(msg, flush=True)
                    if args.out:
                        with open(args.out, "a", encoding="utf-8") as o:
                            o.write(msg + "\n")
                    if args.webhook:
                        _notify(args.webhook, msg)
                continue        # 可能还有更多新数据，立刻再读

            if args.duration and time.time() - started >= args.duration:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止监控", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
