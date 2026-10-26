"""控制面 API 的最小自检：配置生成、参数校验、HTTP 基本流程。

不依赖 DeepStream/GPU；运行:
    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.api.config_store import ConfigStore
from server.api.control_plane import ControlPlaneServer
from server.api.errors import ApiError
from server.api.operations import ApplyWorker, OperationStore
from server.alert.rules import parse_rules


def _config() -> dict:
    return {
        "model": {
            "person_conf_threshold": 0.6,
            "gies": {
                "person": {"kind": "detector", "source": "models/person/yolo26n.pt", "uid": 1},
                "helmet": {"kind": "detector", "source": "models/helmet/best.pt",
                           "uid": 3, "violation": "head"},
                "vest_cls": {"kind": "classifier", "source": "models/vest/best.pt",
                             "uid": 6, "violation": "no_vest"},
            },
        },
        "rules": {
            "no_helmet": {"gie": "helmet", "cooldown_seconds": 10,
                          "min_detection_count": 3, "attribute_threshold": 0.5},
            "no_vest": {"gie": "vest_cls", "cooldown_seconds": 300,
                        "min_detection_count": 15, "attribute_threshold": 0.8},
        },
        "webhook": {"url": "http://example/webhook", "timeout": 10, "retries": 2},
        "cameras": [{
            "id": "c1", "name": "C1", "rtsp_url": "rtsp://u:p@host/s",
            "enabled": True, "active_rules": ["no_helmet"],
        }],
        "source": {"reconnect": {
            "attempts": 10, "timeout": 10, "interval": 10,
            "latency": 500, "stall_seconds": 180,
        }},
    }


class ConfigStoreTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="vi-test-"))
        self.config_path = self.base / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")
        self.store = ConfigStore(self.config_path, self.base / "runtime")

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def test_create_camera_generates_per_camera_rule(self):
        candidate = self.store.create_camera({
            "id": "c2",
            "rtsp_url": "rtsp://u:p@host/s2",
            "algorithms": [
                {"id": "no_helmet", "params": {
                    "cooldown_seconds": 60,
                    "min_detection_count": 5,
                    "attribute_threshold": 0.9,
                }},
                {"id": "no_vest", "params": {
                    "cooldown_seconds": 600,
                    "min_detection_count": 20,
                }},
            ],
        })
        generated = [name for name in candidate["rules"] if name.startswith("__cam__")]
        self.assertEqual(len(generated), 2)
        self.store.write_raw(candidate)
        view = self.store.get_camera("c2")
        self.assertEqual([alg["id"] for alg in view["algorithms"]], ["no_helmet", "no_vest"])
        self.assertEqual(view["algorithms"][0]["params"]["attribute_threshold"], 0.9)
        self.assertEqual(view["algorithms"][1]["params"]["min_detection_count"], 20)

        parsed = parse_rules(candidate["rules"])
        generated_rules = [rule for name, rule in parsed.items() if name.startswith("__cam__")]
        self.assertTrue(generated_rules)
        self.assertEqual(generated_rules[0].alert_type, "no_helmet")

    def test_classifier_threshold_rejected(self):
        with self.assertRaises(ApiError) as ctx:
            self.store.set_algorithms("c1", {"algorithms": [
                {"id": "no_vest", "params": {"attribute_threshold": 0.9}},
            ]})
        self.assertEqual(ctx.exception.status, 422)

    def test_rtsp_password_masked(self):
        self.assertEqual(self.store.get_camera("c1")["rtsp_url"], "rtsp://u:***@host/s")


class OperationStoreTest(unittest.TestCase):
    def test_single_flight(self):
        base = Path(tempfile.mkdtemp(prefix="vi-ops-test-"))
        try:
            ops = OperationStore(base)
            ops.create("create_camera", "c2", {})
            with self.assertRaises(ApiError) as ctx:
                ops.create("create_camera", "c3", {})
            self.assertEqual(ctx.exception.status, 409)
        finally:
            shutil.rmtree(base, ignore_errors=True)


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="vi-api-test-"))
        self.config_path = self.base / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(_config(), sort_keys=False), encoding="utf-8")
        self.store = ConfigStore(self.config_path, self.base / "runtime")
        self.ops = OperationStore(self.base / "runtime")
        self.worker = ApplyWorker(
            self.store, self.ops, self.base / "runtime" / "status.json", timeout=0.2)
        self.worker.start()
        self.server = ControlPlaneServer(
            ("127.0.0.1", 0), self.store, self.ops, self.worker,
            api_key="test-key",
            status_path=self.base / "runtime" / "status.json",
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.worker.stop()
        shutil.rmtree(self.base, ignore_errors=True)

    def _request(self, method: str, path: str, body: dict | None = None,
                 key: str | None = None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, method=method)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if key:
            request.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_auth_and_health(self):
        self.assertEqual(self._request("GET", "/healthz")[0], 200)
        self.assertEqual(self._request("GET", "/v1/cameras")[0], 401)
        self.assertEqual(self._request("GET", "/v1/cameras", key="test-key")[0], 200)

    def test_create_camera_returns_operation(self):
        status, payload = self._request("POST", "/v1/cameras", {
            "id": "c2", "rtsp_url": "rtsp://h/s", "algorithms": ["no_helmet"],
        }, key="test-key")
        self.assertEqual(status, 202)
        self.assertIn("operation_id", payload)
        status, operation = self._request(
            "GET", f"/v1/operations/{payload['operation_id']}", key="test-key")
        self.assertEqual(status, 200)
        self.assertNotIn("candidate", operation)

    def test_unknown_algorithm_rejected(self):
        status, payload = self._request("POST", "/v1/cameras", {
            "id": "c3", "rtsp_url": "rtsp://h/s", "algorithms": ["nope"],
        }, key="test-key")
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "ALGORITHM_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
