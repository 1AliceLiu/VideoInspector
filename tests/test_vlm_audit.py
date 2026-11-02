"""vlm-audit 低置信巡检单测：候选筛选（重叠/TTL 去重/上限）与 1 人 3 问判定逻辑。

不需要 DeepStream：帧用假对象，大模型用本机假 HTTP 服务；在容器内运行：
    docker exec -w /tmp/vi-check safety-detector python3 tests/test_vlm_audit.py
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.alert import vlm_audit


def _serve_map(mapping: dict, default: str = "yes"):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            text = json.loads(raw)["messages"][0]["content"][0]["text"]
            ans = next((v for k, v in mapping.items() if k in text), default)
            body = json.dumps({"choices": [{"message": {"content": ans}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1/chat/completions"


class TestSelect(unittest.TestCase):
    def test_exclude_high_and_dedup_ttl(self):
        now = 1000.0
        lows = [((10, 10, 100, 200), 0.5), ((300, 10, 400, 200), 0.4)]
        highs = [(12, 12, 98, 198)]                 # 与第一个框重叠 → 排除
        recent = [((302, 12, 398, 198), now - 10)]  # 与第二个框重叠且在 TTL 内 → 排除
        got = vlm_audit.select_candidates(lows, highs, recent, now, ttl=60)
        self.assertEqual(got, [])

    def test_ttl_expired_allows(self):
        now = 1000.0
        lows = [((300, 10, 400, 200), 0.4)]
        recent = [((302, 12, 398, 198), now - 120)]  # TTL 外 → 可再审
        got = vlm_audit.select_candidates(lows, [], recent, now, ttl=60)
        self.assertEqual(len(got), 1)

    def test_limit_and_conf_order(self):
        lows = [((i * 300, 0, i * 300 + 100, 200), c)
                for i, c in enumerate((0.30, 0.55, 0.40))]
        got = vlm_audit.select_candidates(lows, [], [], 0.0, ttl=60, limit=2)
        self.assertEqual([c for _, c in got], [0.55, 0.40])


class TestReview(unittest.TestCase):
    frame = staticmethod(lambda: np.full((720, 1280, 3), 30, np.uint8))

    def test_violation_mapping(self):
        srv, url = _serve_map({"red box": "yes", "head and shoulders": "no", "torso": "yes"})
        self.addCleanup(srv.shutdown)
        person, helmet, vest, violated, jpegs, ms = vlm_audit.review_person(
            url, "m", self.frame(), (300, 100, 420, 500))
        self.assertEqual((person, helmet, vest), ("yes", "no", "yes"))
        self.assertEqual(violated, ["no_helmet"])       # 只有帽违规
        self.assertTrue(jpegs["person"] and jpegs["head"] and jpegs["torso"])

    def test_non_person_no_violation(self):
        srv, url = _serve_map({"red box": "no"})
        self.addCleanup(srv.shutdown)
        person, helmet, vest, violated, _, _ = vlm_audit.review_person(
            url, "m", self.frame(), (300, 100, 420, 500))
        self.assertEqual((person, helmet, vest), ("no", None, None))
        self.assertEqual(violated, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
