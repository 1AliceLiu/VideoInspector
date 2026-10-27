"""控制面 HTTP API（标准库实现，无额外 Web 框架依赖）。

启动:
    python -m server.api.control_plane \
        --config deploy/config.yaml --runtime-dir deploy/runtime \
        --host 0.0.0.0 --port 8080

只读写 config.yaml / runtime 文件，不直接操作 DeepStream。检测进程通过
runtime/reload.request 收到重载通知后重建管线。
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from server.api.config_store import ConfigStore
from server.api.errors import ApiError
from server.api.operations import ApplyWorker, OperationStore
from server.api.util import now_iso

MAX_BODY = 2 * 1024 * 1024


def _read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


class ControlPlaneServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, store: ConfigStore, operations: OperationStore,
                 worker: ApplyWorker, api_key: str, status_path: Path,
                 allow_no_auth: bool = False):
        super().__init__(address, Handler)
        self.store = store
        self.operations = operations
        self.worker = worker
        self.api_key = api_key
        self.status_path = status_path
        self.allow_no_auth = allow_no_auth


class Handler(BaseHTTPRequestHandler):
    server_version = "VideoInspectorAPI/0.2"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---------------------------------------------------------------- HTTP 入口
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def _dispatch(self, method: str):
        try:
            parsed = urlsplit(self.path)
            parts = [part for part in parsed.path.split("/") if part]
            if parts == ["healthz"]:
                return self._health()
            if not self._authorized():
                raise ApiError(401, "UNAUTHORIZED", "API Key 无效")
            if not parts or parts[0] != "v1":
                raise ApiError(404, "NOT_FOUND", "unknown path")
            return self._route(method, parts, parse_qs(parsed.query))
        except ApiError as exc:
            self._json(exc.status, {"error": {
                "code": exc.code, "message": exc.message,
                "request_id": self.headers.get("X-Request-Id"),
            }})
        except Exception as exc:
            self._json(500, {"error": {
                "code": "INTERNAL_ERROR", "message": str(exc),
                "request_id": self.headers.get("X-Request-Id"),
            }})

    def _route(self, method: str, parts: list[str], query: dict):
        if parts == ["v1", "algorithms"] and method == "GET":
            return self._json(200, {"items": self.server.store.list_algorithms()})

        if parts == ["v1", "cameras"]:
            if method == "GET":
                return self._list_cameras(query)
            if method == "POST":
                return self._create_camera()

        if len(parts) == 3 and parts[:2] == ["v1", "cameras"]:
            camera_id = parts[2]
            if method == "GET":
                return self._json(200, self.server.store.get_camera(
                    camera_id, self._status_cameras()))
            if method == "PATCH":
                return self._update_camera(camera_id)
            if method == "DELETE":
                return self._delete_camera(camera_id)

        if len(parts) == 4 and parts[:2] == ["v1", "cameras"] and parts[3] == "algorithms":
            if method == "PUT":
                return self._set_algorithms(parts[2])
        if len(parts) == 4 and parts[:2] == ["v1", "cameras"] and parts[3] == "probe":
            if method == "POST":
                return self._probe(parts[2])

        if len(parts) == 3 and parts[:2] == ["v1", "operations"] and method == "GET":
            op = self.server.operations.get(parts[2])
            return self._json(200, self._operation_view(op))

        raise ApiError(404, "NOT_FOUND", f"unknown route: {method} {self.path}")

    # ---------------------------------------------------------------- 路由实现
    def _health(self):
        status = _read_status(self.server.status_path)
        self._json(200, {
            "status": "ok",
            "pipeline": status.get("pipeline", "unknown"),
            "config_revision": self.server.store.revision(),
        })

    def _list_cameras(self, query: dict):
        status = self._status_cameras()
        items = self.server.store.list_cameras(status)
        enabled = (query.get("enabled") or [None])[0]
        algorithm = (query.get("algorithm") or [None])[0]
        if enabled is not None:
            want = enabled.lower() in ("1", "true", "yes")
            items = [item for item in items if item["enabled"] == want]
        if algorithm:
            items = [item for item in items
                     if any(alg["id"] == algorithm for alg in item["algorithms"])]
        try:
            page = max(int((query.get("page") or ["1"])[0]), 1)
            size = min(max(int((query.get("page_size") or ["50"])[0]), 1), 200)
        except ValueError:
            raise ApiError(400, "INVALID_ARGUMENT", "page/page_size 必须是整数")
        total = len(items)
        start = (page - 1) * size
        self._json(200, {"items": items[start:start + size], "total": total})

    def _create_camera(self):
        body = self._body()
        existing = self._idempotent()
        if existing:
            return self._json(202, self._operation_view(existing))
        candidate = self.server.store.create_camera(body)
        op = self.server.operations.create(
            "create_camera", body.get("id"), candidate,
            self.headers.get("Idempotency-Key"),
        )
        self.server.worker.submit(op)
        self._json(202, self._operation_view(op))

    def _update_camera(self, camera_id: str):
        existing = self._idempotent()
        if existing:
            return self._json(202, self._operation_view(existing))
        candidate = self.server.store.update_camera(camera_id, self._body())
        op = self.server.operations.create(
            "update_camera", camera_id, candidate,
            self.headers.get("Idempotency-Key"),
        )
        self.server.worker.submit(op)
        self._json(202, self._operation_view(op))

    def _delete_camera(self, camera_id: str):
        existing = self._idempotent()
        if existing:
            return self._json(202, self._operation_view(existing))
        candidate = self.server.store.delete_camera(camera_id)
        op = self.server.operations.create(
            "delete_camera", camera_id, candidate,
            self.headers.get("Idempotency-Key"),
        )
        self.server.worker.submit(op)
        self._json(202, self._operation_view(op))

    def _set_algorithms(self, camera_id: str):
        existing = self._idempotent()
        if existing:
            return self._json(202, self._operation_view(existing))
        candidate = self.server.store.set_algorithms(camera_id, self._body())
        op = self.server.operations.create(
            "bind_algorithms", camera_id, candidate,
            self.headers.get("Idempotency-Key"),
        )
        self.server.worker.submit(op)
        self._json(202, self._operation_view(op))

    def _probe(self, camera_id: str):
        existing = self._idempotent()
        if existing:
            return self._json(202, self._operation_view(existing))
        # 确认摄像头存在；探测本身在 worker 里跑。
        self.server.store.get_camera(camera_id)
        op = self.server.operations.create(
            "probe_camera", camera_id, None,
            self.headers.get("Idempotency-Key"),
        )
        self.server.worker.submit(op)
        self._json(202, self._operation_view(op))

    # ---------------------------------------------------------------- 工具
    def _authorized(self) -> bool:
        if self.server.allow_no_auth:
            return True
        expected = self.server.api_key
        if not expected:
            return False
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        return hmac.compare_digest(header[7:], expected)

    def _idempotent(self) -> dict | None:
        return self.server.operations.find_idempotent(
            self.headers.get("Idempotency-Key"))

    @staticmethod
    def _operation_view(op: dict) -> dict:
        return {
            "operation_id": op["operation_id"],
            "action": op.get("action"),
            "camera_id": op.get("camera_id"),
            "status": op.get("status", "pending"),
            "config_revision": op.get("config_revision"),
            "created_at": op.get("created_at"),
            "finished_at": op.get("finished_at"),
            "result": op.get("result"),
            "error": op.get("error"),
        }

    def _status_cameras(self) -> dict:
        status = _read_status(self.server.status_path)
        return {str(cam.get("id")): cam for cam in status.get("cameras") or []}

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ApiError(400, "INVALID_ARGUMENT", "请求体不能为空")
        if length > MAX_BODY:
            raise ApiError(413, "PAYLOAD_TOO_LARGE", "请求体超过 2MB")
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            raise ApiError(400, "INVALID_JSON", f"JSON 解析失败: {exc}") from exc
        if not isinstance(data, dict):
            raise ApiError(400, "INVALID_ARGUMENT", "请求体必须是 JSON 对象")
        return data

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="VideoInspector 控制面 API")
    parser.add_argument(
        "--config", "-c",
        default=os.environ.get("SAFETY_CONFIG", "/app/deploy/config/config.yaml"),
    )
    parser.add_argument(
        "--runtime-dir",
        default=os.environ.get("SAFETY_RUNTIME_DIR", "/app/deploy/runtime"),
    )
    parser.add_argument("--host", default=os.environ.get("SAFETY_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("SAFETY_API_PORT", "8080")))
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    if not config_path.exists():
        sys.exit(f"配置文件不存在: {config_path}")

    store = ConfigStore(config_path, runtime_dir)
    operations = OperationStore(runtime_dir)
    operations.mark_stale()
    timeout = float(os.environ.get("SAFETY_APPLY_TIMEOUT", "90"))
    worker = ApplyWorker(store, operations, runtime_dir / "status.json", timeout=timeout)
    worker.start()

    api_key = os.environ.get("SAFETY_API_KEY", "")
    allow_no_auth = os.environ.get("SAFETY_API_ALLOW_NO_AUTH", "0") == "1"
    if not api_key and not allow_no_auth:
        print("⚠ 未设置 SAFETY_API_KEY：所有写接口将返回 401。"
              "本地调试可设 SAFETY_API_ALLOW_NO_AUTH=1", file=sys.stderr)

    server = ControlPlaneServer(
        (args.host, args.port), store, operations, worker,
        api_key=api_key, status_path=runtime_dir / "status.json",
        allow_no_auth=allow_no_auth,
    )

    def _shutdown(signum, frame):
        print(f"收到信号 {signum}，正在停止 API...", file=sys.stderr)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    print(f"控制面 API 启动: http://{args.host}:{args.port} "
          f"(config={config_path}, runtime={runtime_dir})", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        worker.stop()
        server.server_close()


if __name__ == "__main__":
    main()
