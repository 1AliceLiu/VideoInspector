"""Webhook 推送单测：平台业务拒绝（HTTP 200 + body.status）与重试行为。

不依赖 DeepStream/GPU；本机起一个临时 HTTP 服务模拟平台响应。
运行:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.alert.webhook import WebhookAlerter


def _serve(status: int, body: str):
    """启动返回固定 (status, body) 的本机服务，返回 (url, 请求计数列表)。"""
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            hits.append(self.path)
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/webhook"
    return server, url, hits


class TestWebhookAlerter(unittest.TestCase):
    PAYLOAD = {"event_id": "evt_test", "alert_type": "intrusion"}

    def _run(self, status: int, body: str, retries: int = 2) -> tuple[bool, list[str]]:
        server, url, hits = _serve(status, body)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        alerter = WebhookAlerter(url, timeout=5, retries=retries)
        return alerter.send(dict(self.PAYLOAD)), hits

    def test_success_envelope(self):
        ok, hits = self._run(200, '{"status":200,"msg":"ok","res":null}')
        self.assertTrue(ok)
        self.assertEqual(len(hits), 1)

    def test_plain_2xx_body_is_success(self):
        ok, hits = self._run(200, "OK")
        self.assertTrue(ok)
        self.assertEqual(len(hits), 1)

    def test_business_rejection_is_failure_without_retry(self):
        ok, hits = self._run(200, '{"status":404,"msg":"不支持的告警类型: intrusion","res":null}')
        self.assertFalse(ok)
        self.assertEqual(len(hits), 1)

    def test_http_error_is_retried(self):
        ok, hits = self._run(500, "boom", retries=2)
        self.assertFalse(ok)
        self.assertEqual(len(hits), 3)


if __name__ == "__main__":
    unittest.main()
