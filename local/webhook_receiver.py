#!/usr/bin/env python3
"""
Webhook 接收器 — 模拟第三方消费端 + 本地告警看板
- 接收检测服务端推送的告警（含 alert_type 规则名，如 no_vest/no_helmet/intrusion 及 base64 证据帧）
- 解码并保存证据帧到本地磁盘，帧文件名 <timestamp>_<alert_type>.jpg
- 推送数据 → JSONL；运维日志 → loguru
- **浏览器看板**：GET / 显示最近告警（证据图 + 类型 + 摄像头 + 命中区域），3 秒自动刷新

启动方式:
    python local/webhook_receiver.py
    python local/webhook_receiver.py --port 8888
    python local/webhook_receiver.py --save-dir evidence
看板地址:
    http://<主机>:<端口>/          （图片走 /frames/<摄像头>/<文件>，仅允许 save-dir 内路径）
"""

import argparse
import base64
import html
import json
import sys
import urllib.parse
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from loguru import logger


# ── 模块级状态 ──────────────────────────────────────────────────────────
_receive_count = 0
_save_dir: Path | None = None
_payload_log: Path | None = None
_PAGE_RECENT = 60        # 看板最多显示最近多少条
_REFRESH_SECONDS = 3


# ── 看板 ────────────────────────────────────────────────────────────────
def _read_recent_records(limit: int = _PAGE_RECENT) -> list[dict]:
    """倒序读 payload.jsonl 最近若干条（文件小，直接全读；不必建索引）。"""
    if not _payload_log or not _payload_log.exists():
        return []
    out: list[dict] = []
    try:
        with _payload_log.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return list(reversed(out[-limit:]))


def _frame_url(saved_to: str | None) -> str | None:
    """把落盘路径转成 /frames/... URL；越界或不在 save-dir 内一律不给出。"""
    if not saved_to or not _save_dir:
        return None
    try:
        p = Path(saved_to).resolve()
        p.relative_to(_save_dir.resolve())
    except (ValueError, OSError):
        return None
    return "/frames/" + urllib.parse.quote(p.relative_to(_save_dir).as_posix())


def render_page() -> str:
    records = _read_recent_records()
    cards = []
    for r in records:
        atype = html.escape(str(r.get("alert_type", "?")))
        cam = html.escape(str(r.get("camera_name") or r.get("camera_id") or "?"))
        cid = html.escape(str(r.get("camera_id", "")))
        ts = html.escape(str(r.get("timestamp") or r.get("received_at") or ""))
        zones = r.get("zones") or []
        zones_txt = html.escape(", ".join(str(z) for z in zones)) if zones else "全图"
        n_obj = len(r.get("objects") or [])
        url = _frame_url(r.get("frame_saved_to"))
        img = (f'<a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a>'
               if url else '<div class="noimg">无证据帧</div>')
        cards.append(
            f'<figure><div class="ph">{img}</div><figcaption>'
            f'<b class="t t{atype}">{atype}</b>'
            f'<span class="m">{cam} · {cid}</span>'
            f'<span class="m">{ts}</span>'
            f'<span class="z">区域: {zones_txt}</span>'
            f'<span class="m">目标 {n_obj}</span>'
            f'</figcaption></figure>'
        )
    body = "".join(cards) or '<p class="empty">还没有告警。等有人触发规则，页面会自动刷新。</p>'
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{_REFRESH_SECONDS}">
<title>告警看板</title>
<style>
 body{{margin:0;background:#111;color:#ddd;font:14px/1.5 system-ui,sans-serif}}
 header{{padding:14px 18px;background:#1b1b1b;border-bottom:1px solid #2b2b2b;
        display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}}
 header h1{{font-size:16px;margin:0}}
 header .s{{color:#8a8a8a}}
 main{{padding:16px;display:grid;gap:14px;
      grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}}
 figure{{margin:0;background:#1b1b1b;border:1px solid #2b2b2b;border-radius:6px;overflow:hidden}}
 .ph{{height:190px;background:#000;display:flex;align-items:center;justify-content:center}}
 .ph img{{width:100%;height:100%;object-fit:cover;display:block}}
 .noimg{{color:#666}}
 figcaption{{padding:8px 10px;display:flex;flex-direction:column;gap:2px}}
 .t{{font-size:15px}} .tintrusion{{color:#ffb300}} .tno_helmet{{color:#ff5b5b}}
 .tno_vest{{color:#4da3ff}} .tsmoking{{color:#b07cff}}
 .m{{color:#8a8a8a;font-size:12px}} .z{{color:#7fd47f;font-size:12px}}
 .empty{{color:#8a8a8a;padding:24px}}
</style></head><body>
<header><h1>告警看板</h1>
<span class="s">已接收 { _receive_count } 条（本进程）· 共 { len(records) } 条落盘记录
· 每 {_REFRESH_SECONDS}s 自动刷新 · 图片为服务端 OSD 后补画的证据帧</span></header>
<main>{body}</main>
</body></html>"""


def _resolve_frame(path: str) -> Path | None:
    """GET /frames/<...> → 真实文件；任何跳出 save-dir 的企图都返回 None。"""
    if not _save_dir:
        return None
    rel = urllib.parse.unquote(path[len("/frames/"):])
    if not rel or rel.startswith("/") or ".." in Path(rel).parts:
        return None
    try:
        p = (_save_dir / rel).resolve()
        p.relative_to(_save_dir.resolve())
    except (ValueError, OSError):
        return None
    return p if p.is_file() else None


# ── Handler ──────────────────────────────────────────────────────────────
class WebhookHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        logger.debug("HTTP {} {}", args[0], self.client_address[0])

    def do_POST(self):
        global _receive_count

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("收到非法 JSON，来源: {}", self.client_address[0])
            self._respond(400, {"error": "Invalid JSON"})
            return

        _receive_count += 1
        count = _receive_count

        # 立即响应，避免磁盘 I/O / 日志输出阻塞发送端
        self._respond(200, {"status": "ok", "received": count})

        # ── 以下操作在 HTTP 响应之后执行，不影响发送端 ──

        # 1. 解码并保存证据帧（base64 主动剥离，避免 JSONL 膨胀）
        frame_saved_to = None
        frame_b64 = payload.pop("frame_base64", None)
        if frame_b64 and _save_dir:
            try:
                camera_id = payload.get("camera_id", "unknown")
                alert_type = payload.get("alert_type") or "unknown"
                ts = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                ts_str = ts[:19].replace(":", "").replace("T", "_")
                frame_dir = _save_dir / camera_id
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / f"{ts_str}_{alert_type}.jpg"
                frame_path.write_bytes(base64.b64decode(frame_b64))
                frame_saved_to = str(frame_path)
                logger.info("帧已保存: {}", frame_path)
            except Exception:
                logger.exception("解码/保存帧失败")

        # 2. 推送数据追加到 JSONL（不含 frame_base64，避免日志膨胀）
        if _payload_log:
            record = {
                "received_at": datetime.now(timezone.utc).isoformat(),
                **payload,
                "frame_saved_to": frame_saved_to,
            }
            with open(_payload_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        # 3. 控制台摘要（通过 logger，无阻塞 flush）
        self._log_summary(count, payload, frame_saved_to)

    def do_GET(self):
        """看板首页 / 证据图 / 兼容旧的 JSON 状态查询（Accept: application/json 或非浏览器）。"""
        path = self.path.split("?", 1)[0]
        if path.startswith("/frames/"):
            f = _resolve_frame(path)
            if f is None:
                self._respond(404, {"error": "not found"})
                return
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "max-age=60")
            self.end_headers()
            self.wfile.write(data)
            return

        wants_html = path in ("/", "/index.html") and "json" not in self.headers.get("Accept", "").lower()
        if wants_html:
            body = render_page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self._respond(200, {"service": "Webhook Receiver", "received_count": _receive_count})

    def _respond(self, status, data):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log_summary(self, count, payload, frame_path):
        """通过 logger 输出告警摘要，无阻塞 flush。"""
        objects = payload.get("objects", [])  # 修复: 键名为 "objects" 而非 "detections"
        zones = payload.get("zones") or []
        logger.info(
            "Alert #{} | Type: {} | Camera: {} ({}) | Objects: {} | Zones: {} | Frame: {}",
            count,
            payload.get("alert_type", "?"),
            payload.get("camera_name", "?"),
            payload.get("camera_id", "?"),
            len(objects),
            ",".join(zones) or "-",
            frame_path or "N/A",
        )
        for i, d in enumerate(objects[:10]):
            bbox = d.get("bbox", [])
            logger.debug(
                "  [{}/{}] {} conf={:.2f} bbox={}",
                i + 1, len(objects), d.get("class", "?"),
                d.get("confidence", 0), bbox,
            )
        if len(objects) > 10:
            logger.debug("  ... and {} more", len(objects) - 10)


# ── 入口 ──────────────────────────────────────────────────────────────────
def main():
    global _save_dir, _payload_log

    parser = argparse.ArgumentParser(description="Webhook 接收器")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=9999)
    parser.add_argument("--save-dir", default="alerts")
    args = parser.parse_args()

    _save_dir = Path(args.save_dir)
    _save_dir.mkdir(parents=True, exist_ok=True)
    _payload_log = _save_dir / "payload.jsonl"

    # loguru：控制台
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    # ThreadingHTTPServer：看板会并发请求多张证据图，单线程模型会在取图期间
    # 挡住检测服务端的告警 POST（响应变慢 → 触发对端 timeout/重试）。
    server = ThreadingHTTPServer((args.host, args.port), WebhookHandler)
    logger.info("Receiver 启动 — {}:{}, 目录 {}", args.host, args.port, args.save_dir)
    logger.info("看板地址 — http://{}:{}",
                "127.0.0.1" if args.host == "0.0.0.0" else args.host, args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        logger.info("Receiver 已停止 — 共 {} 条", _receive_count)


if __name__ == "__main__":
    main()
