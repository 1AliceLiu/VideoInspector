"""A/B 多节点聚合 API 测试（B2：读合并 + 写路由）。"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.api.control_plane import ControlPlaneServer
from server.api.errors import ApiError
from server.api.nodes import NodeRegistry
from server.api.operations import ApplyWorker, OperationStore
from server.api.util import now_iso


def _base_config(camera_id: str, name: str, active_rules: list[str]) -> dict:
    return {
        "model": {
            "person_conf_threshold": 0.6,
            "gies": {
                "person": {"kind": "detector", "source": "models/person.pt", "uid": 1},
                "helmet": {"kind": "detector", "source": "models/helmet.pt",
                           "uid": 3, "violation": "head"},
                "vest_cls": {"kind": "classifier", "source": "models/vest.pt",
                             "uid": 6, "violation": "no_vest"},
                "smoke": {"kind": "detector", "attach": "person",
                          "source": "models/smoke.pt", "uid": 7,
                          "violation": "cigarette"},
            },
        },
        "rules": {
            "no_helmet": {"gie": "helmet", "cooldown_seconds": 10,
                          "min_detection_count": 3, "attribute_threshold": 0.5},
            "no_vest": {"gie": "vest_cls", "cooldown_seconds": 300,
                        "min_detection_count": 15, "attribute_threshold": 0.8},
            "smoking": {"gie": "smoke", "cooldown_seconds": 30,
                        "min_detection_count": 3, "attribute_threshold": 0.75},
        },
        "webhook": {"url": "http://example/webhook", "timeout": 10, "retries": 2},
        "cameras": [{
            "id": camera_id,
            "name": name,
            "rtsp_url": f"rtsp://u:p@host/{camera_id}",
            "enabled": True,
            "active_rules": active_rules,
        }],
        "source": {"reconnect": {
            "attempts": 10, "timeout": 10, "interval": 10,
            "latency": 500, "stall_seconds": 180,
        }},
    }


def _write_config(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


class MultiNodeApiTest(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="vi-multi-test-"))
        self.config_a = self.base / "config.a.yaml"
        self.config_b = self.base / "config.b.yaml"
        self.runtime_a = self.base / "runtime-a"
        self.runtime_b = self.base / "runtime-b"
        _write_config(self.config_a, _base_config("1195", "1195", ["no_helmet", "no_vest"]))
        _write_config(self.config_b, _base_config("720", "720", ["smoking"]))

        (self.runtime_a / "status.json").parent.mkdir(parents=True, exist_ok=True)
        (self.runtime_b / "status.json").parent.mkdir(parents=True, exist_ok=True)
        (self.runtime_a / "status.json").write_text(json.dumps({
            "pipeline": "running", "config_revision": 3,
            "cameras": [{"id": "1195", "online": True}],
        }), encoding="utf-8")
        (self.runtime_b / "status.json").write_text(json.dumps({
            "pipeline": "running", "config_revision": 4,
            "cameras": [{"id": "720", "online": True}],
        }), encoding="utf-8")

        self.registry = NodeRegistry.from_paths({
            "a": (self.config_a, self.runtime_a),
            "b": (self.config_b, self.runtime_b),
        })
        self.registry.validate()

        # 测试里不跑真实 detector：让 ApplyWorker 直接认为重载成功。
        self.wait_patcher = patch(
            "server.api.operations._wait_for_status", return_value=True)
        self.wait_patcher.start()

        self.ops = OperationStore(self.base / "runtime-api")
        self.ops.mark_stale()
        self.workers = {}
        for node in self.registry.nodes.values():
            worker = ApplyWorker(
                node.store, self.ops, node.status_path, timeout=0.5)
            worker.start()
            self.workers[node.name] = worker

        self.server = ControlPlaneServer(
            ("127.0.0.1", 0), api_key="test-key",
            registry=self.registry, operations=self.ops, workers=self.workers,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        for worker in self.workers.values():
            worker.stop()
        self.wait_patcher.stop()
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

    def _wait_op(self, operation_id: str, timeout: float = 3.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            op = self.ops.get(operation_id)
            if op.get("status") in ("succeeded", "failed", "rolled_back"):
                return op
            time.sleep(0.02)
        self.fail(f"operation {operation_id} 未在 {timeout}s 内结束")

    def test_list_merges_nodes_and_counts(self):
        status, payload = self._request("GET", "/v1/cameras", key="test-key")
        self.assertEqual(status, 200)
        self.assertEqual(payload["total"], 2)
        by_id = {item["id"]: item for item in payload["items"]}
        self.assertEqual(by_id["1195"]["node"], "a")
        self.assertEqual(by_id["1195"]["algorithm_count"], 2)
        self.assertEqual([a["id"] for a in by_id["1195"]["algorithms"]],
                         ["no_helmet", "no_vest"])
        self.assertEqual(by_id["720"]["node"], "b")
        self.assertEqual(by_id["720"]["algorithm_count"], 1)

    def test_get_camera_finds_node(self):
        status, payload = self._request("GET", "/v1/cameras/1195", key="test-key")
        self.assertEqual(status, 200)
        self.assertEqual(payload["node"], "a")
        self.assertEqual(payload["algorithm_count"], 2)
        status, payload = self._request("GET", "/v1/cameras/nope", key="test-key")
        self.assertEqual(status, 404)

    def test_algorithms_global_catalog(self):
        status, payload = self._request("GET", "/v1/algorithms", key="test-key")
        self.assertEqual(status, 200)
        ids = {item["id"] for item in payload["items"]}
        self.assertEqual(ids, {"no_helmet", "no_vest", "smoking"})
        self.assertNotIn("total", payload)

    def test_health_has_nodes(self):
        status, payload = self._request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(payload["nodes"]["a"]["pipeline"], "running")
        self.assertEqual(payload["nodes"]["b"]["pipeline"], "running")
        self.assertEqual(payload["pipeline"], "running")

    def test_auth_required(self):
        self.assertEqual(self._request("GET", "/v1/cameras")[0], 401)

    def test_append_routes_to_a(self):
        status, payload = self._request(
            "POST", "/v1/cameras/1195/algorithms",
            {"algorithms": [{"id": "smoking", "params": {
                "cooldown_seconds": 60}}]}, key="test-key")
        self.assertEqual(status, 202)
        op = self._wait_op(payload["operation_id"])
        self.assertEqual(op["status"], "succeeded")
        self.assertEqual(op["node"], "a")

        status, payload = self._request("GET", "/v1/cameras/1195", key="test-key")
        self.assertEqual(status, 200)
        self.assertEqual(payload["node"], "a")
        self.assertEqual(payload["algorithm_count"], 3)
        self.assertEqual([a["id"] for a in payload["algorithms"]],
                         ["no_helmet", "no_vest", "smoking"])
        self.assertEqual(payload["algorithms"][2]["params"]["cooldown_seconds"], 60)

        a_cam = yaml.safe_load(self.config_a.read_text(encoding="utf-8"))["cameras"][0]
        self.assertEqual(a_cam["active_rules"][:2], ["no_helmet", "no_vest"])
        self.assertTrue(a_cam["active_rules"][2].startswith("__cam__1195"))
        b_cam = yaml.safe_load(self.config_b.read_text(encoding="utf-8"))["cameras"][0]
        self.assertEqual(b_cam["id"], "720")
        self.assertEqual(b_cam["active_rules"], ["smoking"])

    def test_append_routes_to_b(self):
        status, payload = self._request(
            "POST", "/v1/cameras/720/algorithms",
            {"algorithms": [{"id": "no_helmet"}]}, key="test-key")
        self.assertEqual(status, 202)
        op = self._wait_op(payload["operation_id"])
        self.assertEqual(op["status"], "succeeded")
        self.assertEqual(op["node"], "b")

        status, payload = self._request("GET", "/v1/cameras/720", key="test-key")
        self.assertEqual(status, 200)
        self.assertEqual(payload["node"], "b")
        self.assertEqual(payload["algorithm_count"], 2)
        self.assertIn("no_helmet", [a["id"] for a in payload["algorithms"]])

    def test_append_duplicate_rejected(self):
        status, payload = self._request(
            "POST", "/v1/cameras/1195/algorithms",
            {"algorithms": [{"id": "no_vest"}]}, key="test-key")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "ALGORITHM_ALREADY_BOUND")

    def test_create_auto_balances_nodes(self):
        status, payload = self._request("POST", "/v1/cameras", {
            "id": "1300", "name": "1300", "rtsp_url": "rtsp://h/1300",
            "algorithms": ["no_helmet"],
        }, key="test-key")
        self.assertEqual(status, 202)
        op = self._wait_op(payload["operation_id"])
        self.assertEqual(op["status"], "succeeded")
        self.assertEqual(op["node"], "a")

        status, payload = self._request("GET", "/v1/cameras/1300", key="test-key")
        self.assertEqual(status, 200)
        self.assertEqual(payload["node"], "a")

        # A 现在 2 路，B 1 路，下一路应自动放 B。
        status, payload = self._request("POST", "/v1/cameras", {
            "id": "1301", "name": "1301", "rtsp_url": "rtsp://h/1301",
            "algorithms": ["no_helmet"],
        }, key="test-key")
        self.assertEqual(status, 202)
        op = self._wait_op(payload["operation_id"])
        self.assertEqual(op["status"], "succeeded")
        self.assertEqual(op["node"], "b")

        status, payload = self._request("GET", "/v1/cameras/1301", key="test-key")
        self.assertEqual(status, 200)
        self.assertEqual(payload["node"], "b")

    def test_create_explicit_node(self):
        status, payload = self._request("POST", "/v1/cameras", {
            "id": "1302", "name": "1302", "rtsp_url": "rtsp://h/1302",
            "algorithms": ["no_helmet"], "node": "b",
        }, key="test-key")
        self.assertEqual(status, 202)
        op = self._wait_op(payload["operation_id"])
        self.assertEqual(op["status"], "succeeded")
        self.assertEqual(op["node"], "b")

        status, payload = self._request("GET", "/v1/cameras/1302", key="test-key")
        self.assertEqual(status, 200)
        self.assertEqual(payload["node"], "b")

    def test_create_duplicate_camera_rejected(self):
        status, payload = self._request("POST", "/v1/cameras", {
            "id": "1195", "rtsp_url": "rtsp://h/dup", "algorithms": ["no_helmet"],
        }, key="test-key")
        self.assertEqual(status, 409)
        self.assertEqual(payload["error"]["code"], "CAMERA_EXISTS")

    def test_capacity_enforced(self):
        config_a = _base_config("a1", "a1", ["no_helmet"])
        config_b = _base_config("b1", "b1", ["no_helmet"])
        _write_config(self.config_a, config_a)
        _write_config(self.config_b, config_b)
        registry = NodeRegistry.from_paths({
            "a": (self.config_a, self.runtime_a),
            "b": (self.config_b, self.runtime_b),
        }, max_cameras_per_node=1)

        with self.assertRaises(ApiError) as ctx:
            registry.route_create({"id": "new", "rtsp_url": "rtsp://h/new"})
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(ctx.exception.code, "CAPACITY_EXCEEDED")

        with self.assertRaises(ApiError) as ctx:
            registry.route_create({"id": "new", "node": "a"})
        self.assertEqual(ctx.exception.code, "CAPACITY_EXCEEDED")

        # 停用的摄像头不占 enabled 上限，允许先建后启用。
        node = registry.route_create({"id": "new", "enabled": False})
        self.assertIn(node.name, ("a", "b"))

    def test_enable_capacity_enforced(self):
        config_a = _base_config("a1", "a1", ["no_helmet"])
        disabled = _base_config("a2", "a2", ["no_helmet"])["cameras"][0]
        disabled["enabled"] = False
        config_a["cameras"].append(disabled)
        _write_config(self.config_a, config_a)

        registry = NodeRegistry.from_paths({
            "a": (self.config_a, self.runtime_a),
            "b": (self.config_b, self.runtime_b),
        }, max_cameras_per_node=1)
        node_a = registry.nodes["a"]
        registry.ensure_enable_capacity(node_a, "a1")
        with self.assertRaises(ApiError) as ctx:
            registry.ensure_enable_capacity(node_a, "a2")
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(ctx.exception.code, "CAPACITY_EXCEEDED")

    def test_global_single_flight(self):
        op = self.ops.create("create_camera", "x", {}, node="a")
        with self.assertRaises(ApiError) as ctx:
            self.ops.create("create_camera", "y", {}, node="b")
        self.assertEqual(ctx.exception.status, 409)
        self.assertEqual(ctx.exception.code, "OPERATION_IN_PROGRESS")
        self.ops.update(op, status="succeeded", finished_at=now_iso())

    def test_duplicate_camera_rejected(self):
        _write_config(self.config_b, _base_config("1195", "dup", ["smoking"]))
        registry = NodeRegistry.from_paths({
            "a": (self.config_a, self.runtime_a),
            "b": (self.config_b, self.runtime_b),
        })
        with self.assertRaises(ValueError):
            registry.validate()

    def test_mismatched_algorithms_rejected(self):
        config = _base_config("720", "720", ["smoking"])
        config["rules"]["smoking"]["cooldown_seconds"] = 99
        _write_config(self.config_b, config)
        registry = NodeRegistry.from_paths({
            "a": (self.config_a, self.runtime_a),
            "b": (self.config_b, self.runtime_b),
        })
        with self.assertRaises(ApiError):
            registry.validate()

    def test_per_node_camera_cap(self):
        config = _base_config("1195", "1195", ["no_helmet"])
        config["cameras"] = [{
            "id": f"cam{i}",
            "name": f"cam{i}",
            "rtsp_url": f"rtsp://host/{i}",
            "enabled": True,
            "active_rules": ["no_helmet"],
        } for i in range(33)]
        _write_config(self.config_a, config)
        registry = NodeRegistry.from_paths({
            "a": (self.config_a, self.runtime_a),
            "b": (self.config_b, self.runtime_b),
        }, max_cameras_per_node=32)
        with self.assertRaises(ValueError):
            registry.validate()


if __name__ == "__main__":
    unittest.main()
