"""VLM 复核单测：连拍采样、四宫格拼接、多规则 yes/no 门控（含极性）、shadow、fail_open。

不依赖 DeepStream/GPU：帧缓存用假对象（bgr() 返回 numpy），大模型用本机临时
HTTP 服务模拟 OpenAI 兼容 /v1/chat/completions。需要 cv2/numpy/simplejpeg，
在容器内运行：
    docker cp server tests → 容器后: python3 tests/test_vlm_review.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.alert import vlm_review
from server.alert.manager import AlertManager
from server.alert.rules import parse_rules
from server.metadata import AttributeMeta, ObjectMeta

VLM_BASE = {"enabled": True, "rules": ["smoking"],
            "sample_ms": 0, "deadline_ms": 500, "timeout": 5}
OBJ = ObjectMeta("person", 0.9, (300, 100, 420, 500),
                 (AttributeMeta("smoking", 0.6),))


class _Snap:
    """假 EvidenceFrame：固定尺寸 BGR，像素值随帧号变化（便于验证采集到了不同帧）。"""

    def __init__(self, fn):
        self._img = np.full((720, 1280, 3), fn % 200, np.uint8)

    def bgr(self):
        return self._img


class _Cache:
    """每次调用返回递增帧号的新帧（模拟连拍）。"""

    def __init__(self):
        self.n = 0

    def latest_with_meta(self, sid):
        self.n += 1
        return _Snap(self.n), self.n, 0


class _FakeWebhook:
    def __init__(self):
        self.payloads = []

    def send(self, payload):
        self.payloads.append(payload)
        return True


def _serve_answer(answer: str):
    """固定回答 answer 的假 OpenAI chat completions 服务。"""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"choices": [{"message": {"content": answer}}]}).encode()
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


def _manager(url: str, **vlm_extra):
    rules = parse_rules({"smoking": {"gie": "smoke", "cooldown_seconds": 30,
                                     "min_detection_count": 3,
                                     "attribute_threshold": 0.5}})
    cfg = dict(VLM_BASE, url=url, **vlm_extra)
    wh = _FakeWebhook()
    m = AlertManager(camera_id="t1", camera_name="test-cam",
                     rules=list(rules.values()), webhook=wh,
                     vlm=cfg, frame_cache=_Cache(), source_id=0)
    return m, wh


class TestVlmReview(unittest.TestCase):
    snapshot = staticmethod(lambda: np.full((720, 1280, 3), 20, np.uint8))

    def _wait_payload(self, wh, seconds=2.0):
        """_build_and_send 用 daemon 线程推送，轮询等待（复核本身同步调用）。"""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if wh.payloads:
                return wh.payloads[0]
            time.sleep(0.01)
        return None

    def _review(self, m, rule="smoking"):
        m._vlm_review(rule, rule, [OBJ], self.snapshot(), "2026-01-01T00:00:00+00:00")

    def test_yes_sends_with_verdict(self):
        srv, url = _serve_answer("Yes")
        self.addCleanup(srv.shutdown)
        m, wh = _manager(url)
        self._review(m)
        p = self._wait_payload(wh)
        self.assertIsNotNone(p)
        self.assertEqual(p["vlm"]["verdict"], "yes")
        self.assertEqual(p["vlm"]["frames"], 4)
        self.assertIsNotNone(p["frame_base64"])  # 证据帧照常附带

    def test_no_discards(self):
        srv, url = _serve_answer("No.")
        self.addCleanup(srv.shutdown)
        m, wh = _manager(url)
        self._review(m)
        time.sleep(0.2)
        self.assertEqual(wh.payloads, [])

    # --- 多规则：安全帽/反光衣问"是否佩戴"，答 no=违规坐实（极性与吸烟相反）---
    def test_helmet_no_pushes_with_verdict(self):
        srv, url = _serve_answer("No")   # 没戴帽 → 违规坐实 → 推
        self.addCleanup(srv.shutdown)
        m, wh = _manager(url, rules=["no_helmet"])
        self._review(m, "no_helmet")
        p = self._wait_payload(wh)
        self.assertIsNotNone(p)
        self.assertEqual(p["vlm"]["verdict"], "no")   # 下游看原始 verdict，语义由 rule 定
        self.assertEqual(p["vlm"]["frames"], 1)      # 静态属性单帧，不等连拍

    def test_helmet_yes_discards(self):
        srv, url = _serve_answer("Yes")  # 戴了帽 → 误报 → 拦
        self.addCleanup(srv.shutdown)
        m, wh = _manager(url, rules=["no_helmet"])
        self._review(m, "no_helmet")
        time.sleep(0.2)
        self.assertEqual(wh.payloads, [])

    def test_vest_yes_discards(self):
        srv, url = _serve_answer("Yes")  # 穿了反光衣 → 误报 → 拦
        self.addCleanup(srv.shutdown)
        m, wh = _manager(url, rules=["no_vest"])
        self._review(m, "no_vest")
        time.sleep(0.2)
        self.assertEqual(wh.payloads, [])

    def test_unreviewed_rule_pushes_directly(self):
        # 规则在 vlm.rules 内但 REVIEWS 未定义 → 直接推送并记 warning（不误拦）
        srv, url = _serve_answer("No")
        self.addCleanup(srv.shutdown)
        m, wh = _manager(url, rules=["no_harness"])
        self._review(m, "no_harness")
        p = self._wait_payload(wh)
        self.assertIsNotNone(p)
        self.assertNotIn("vlm", p)

    def test_shadow_no_pushes_without_field(self):
        # 观察模式：verdict=no 不拦告警，payload 也不附 vlm 字段（主程序输出不变）
        srv, url = _serve_answer("No.")
        self.addCleanup(srv.shutdown)
        m, wh = _manager(url, shadow=True)
        self._review(m)
        p = self._wait_payload(wh)
        self.assertIsNotNone(p)
        self.assertNotIn("vlm", p)

    def test_unreachable_fail_open(self):
        # 端口 1 无人监听 → ask 返回 None → 默认 fail_open 照推，verdict=skipped
        m, wh = _manager("http://127.0.0.1:1/v1/chat/completions")
        self._review(m)
        p = self._wait_payload(wh)
        self.assertIsNotNone(p)
        self.assertEqual(p["vlm"]["verdict"], "skipped")

    def test_unreachable_fail_closed(self):
        m, wh = _manager("http://127.0.0.1:1/v1/chat/completions", fail_open=False)
        self._review(m)
        time.sleep(0.2)
        self.assertEqual(wh.payloads, [])

    def test_grid_is_2x2_of_distinct_frames(self):
        crops = vlm_review.collect_frames(_Cache(), 0, OBJ.bbox,
                                          frames=4, sample_s=0, deadline_s=2)
        self.assertEqual(len(crops), 4)
        arr = vlm_review.simplejpeg.decode_jpeg(vlm_review.build_grid(crops),
                                                colorspace="BGR")
        self.assertEqual(arr.shape, (vlm_review.TILE * 2, vlm_review.TILE * 2, 3))
        # 四格采样值互不相同（证明取的是 4 张不同帧而非重复帧）
        corners = {int(arr[10, 10, 0]), int(arr[10, 500, 0]),
                   int(arr[500, 10, 0]), int(arr[500, 500, 0])}
        self.assertEqual(len(corners), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
