"""异步变更任务：创建/修改/删除摄像头、绑算法、RTSP 探测。

所有写接口只返回 operation_id，真正的配置替换和管线重载在后台线程执行，
避免 HTTP 请求等 10~30 秒；失败自动回滚上一版配置。
"""

from __future__ import annotations

import hashlib
import json
import queue
import secrets
import threading
import time
from pathlib import Path

from server.api.config_store import ConfigStore
from server.api.errors import ApiError
from server.api.probe import probe_rtsp
from server.api.util import atomic_write_json, now_iso


class OperationStore:
    """operation 落盘：API/容器重启后仍能查到最近结果。"""

    def __init__(self, runtime_dir: str | Path):
        self.runtime_dir = Path(runtime_dir)
        self.dir = self.runtime_dir / "operations"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.idem_dir = self.runtime_dir / "idempotency"
        self.idem_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _new_id(self) -> str:
        return f"op_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"

    def _path(self, operation_id: str) -> Path:
        return self.dir / f"{operation_id}.json"

    def save(self, op: dict) -> None:
        with self._lock:
            atomic_write_json(self._path(op["operation_id"]), op)
            key = op.get("idempotency_key")
            if key:
                digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
                atomic_write_json(self.idem_dir / f"{digest}.json", {
                    "operation_id": op["operation_id"],
                    "created_at": op.get("created_at"),
                })

    def create(self, action: str, camera_id: str | None = None,
               candidate: dict | None = None,
               idempotency_key: str | None = None,
               single_flight: bool = True) -> dict:
        with self._lock:
            if idempotency_key:
                existing = self.find_idempotent(idempotency_key)
                if existing is not None:
                    return existing
            if single_flight and self._has_active():
                raise ApiError(409, "OPERATION_IN_PROGRESS",
                               "已有变更正在生效，请等它完成后再提交")
            op = {
                "operation_id": self._new_id(),
                "action": action,
                "camera_id": camera_id,
                "status": "pending",
                "config_revision": None,
                "created_at": now_iso(),
                "finished_at": None,
                "result": None,
                "error": None,
                "idempotency_key": idempotency_key,
                "candidate": candidate,
            }
            self.save(op)
            return op

    def _has_active(self) -> bool:
        for path in self.dir.glob("op_*.json"):
            try:
                op = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if op.get("status") in ("pending", "applying"):
                return True
        return False

    def get(self, operation_id: str) -> dict:
        path = self._path(operation_id)
        if not path.exists():
            raise ApiError(404, "NOT_FOUND", f"operation {operation_id} not found")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ApiError(500, "OPERATION_CORRUPT", f"operation 文件损坏: {exc}") from exc

    def find_idempotent(self, key: str | None) -> dict | None:
        if not key:
            return None
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        path = self.idem_dir / f"{digest}.json"
        if not path.exists():
            return None
        try:
            operation_id = json.loads(path.read_text(encoding="utf-8"))["operation_id"]
            return self.get(operation_id)
        except Exception:
            return None

    def update(self, op: dict, **fields) -> dict:
        with self._lock:
            op.update(fields)
            self.save(op)
            return op

    def mark_stale(self) -> None:
        """API 重启后，把上次没跑完的任务标记失败，避免永久 pending。"""
        for path in self.dir.glob("op_*.json"):
            try:
                op = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if op.get("status") in ("pending", "applying"):
                op["status"] = "failed"
                op["finished_at"] = now_iso()
                op["error"] = {
                    "code": "API_RESTARTED",
                    "message": "API 进程重启，本次变更未完成；请重新提交",
                }
                self.save(op)


def _wait_for_status(status_path: Path, revision: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            data = json.loads(status_path.read_text(encoding="utf-8"))
            if data.get("config_revision") == revision and data.get("pipeline") == "running":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


class ApplyWorker:
    """单线程串行执行变更，避免并发重启管线。"""

    def __init__(self, store: ConfigStore, operations: OperationStore,
                 status_path: str | Path, timeout: float = 90.0):
        self.store = store
        self.operations = operations
        self.status_path = Path(status_path)
        self.timeout = float(timeout)
        self._queue: queue.Queue[dict] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="api-apply", daemon=True)
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def submit(self, op: dict) -> None:
        self._queue.put(op)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                op = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run(op)
            except Exception as exc:  # 兜底：任何异常都要让 operation 结束
                self.operations.update(
                    op, status="failed", finished_at=now_iso(),
                    error={"code": "INTERNAL_ERROR", "message": str(exc)},
                )

    def _run(self, op: dict) -> None:
        self.operations.update(op, status="applying")

        if op.get("action") == "probe_camera":
            cam = self.store.get_camera_raw(op["camera_id"])
            result = probe_rtsp(cam.get("rtsp_url"), timeout=10)
            result["source_mode"] = cam.get("source_mode")
            self.operations.update(
                op, status="succeeded", finished_at=now_iso(), result=result,
            )
            return

        previous = self.store.read()
        candidate = op.get("candidate")
        if not isinstance(candidate, dict):
            self.operations.update(
                op, status="failed", finished_at=now_iso(),
                error={"code": "INVALID_CANDIDATE", "message": "变更内容为空"},
            )
            return
        if candidate == previous:
            self.operations.update(
                op, status="succeeded", config_revision=self.store.revision(),
                finished_at=now_iso(), error=None,
            )
            return
        backup = self.store.backup(previous)

        try:
            self.store.write_raw(candidate)
            revision = self.store.bump_revision()
            self.store.request_reload(revision)
            ok = _wait_for_status(self.status_path, revision, self.timeout)
        except Exception as exc:
            ok = False
            error = {"code": "APPLY_FAILED", "message": str(exc)}
        else:
            error = {
                "code": "PIPELINE_TIMEOUT",
                "message": f"检测管线 {self.timeout:.0f}s 内未确认新配置，已回滚",
            }

        if ok:
            self.operations.update(
                op, status="succeeded", config_revision=revision,
                finished_at=now_iso(), error=None,
            )
            return

        try:
            rollback_revision = self.store.rollback(backup)
        except Exception as exc:
            rollback_revision = None
            error = {
                "code": "ROLLBACK_FAILED",
                "message": f"回滚失败: {exc}; 原错误: {error['message']}",
            }
        self.operations.update(
            op, status="rolled_back", config_revision=rollback_revision,
            finished_at=now_iso(), error=error,
        )
