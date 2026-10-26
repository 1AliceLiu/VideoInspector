"""检测进程运行状态 → runtime/status.json，供控制面 API 查询。

只读健康计数和告警时间，不参与判定、不阻塞检测线程。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from server.api.util import atomic_write_json, now_iso


class StatusWriter:
    def __init__(self, runtime_dir: str | Path, cameras: list[dict],
                 health, managers: dict | None = None,
                 config_revision: int | str | None = None,
                 interval: float = 5.0):
        self.runtime_dir = Path(runtime_dir)
        self.path = self.runtime_dir / "status.json"
        self.cameras = cameras
        self.health = health
        self.managers = managers or {}
        self.config_revision = config_revision
        try:
            self.config_revision = int(config_revision)
        except (TypeError, ValueError):
            pass
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="api-status", daemon=True)
        self._started_at = now_iso()
        # 保留上一次 status.json 里的最后告警时间，检测进程重启后不丢。
        self._previous_alerts: dict[str, str | None] = {}
        try:
            import json
            old = json.loads(self.path.read_text(encoding="utf-8"))
            for cam in old.get("cameras") or []:
                self._previous_alerts[str(cam.get("id"))] = cam.get("last_alert_at")
        except Exception:
            pass

    def start(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        previous_frames: dict[int, int] = {}
        previous_at = time.monotonic()
        while not self._stop.wait(self.interval):
            now = time.monotonic()
            elapsed = max(now - previous_at, 1e-6)
            cameras = []
            try:
                stats = self.health.stats() if self.health is not None else {}
            except Exception:
                stats = {}
            for index, cam in enumerate(self.cameras):
                item = stats.get(index) or {}
                frames = int(item.get("frames", 0))
                fps = (frames - previous_frames.get(index, frames)) / elapsed
                previous_frames[index] = frames
                age = item.get("last_frame_age")
                manager = self.managers.get(index)
                camera_id = str(cam.get("id"))
                last_alert = getattr(manager, "last_alert_at", None)
                if last_alert is None:
                    last_alert = self._previous_alerts.get(camera_id)
                else:
                    self._previous_alerts[camera_id] = last_alert
                cameras.append({
                    "id": camera_id,
                    "online": bool(age is not None and age <= 30),
                    "last_frame_age_seconds": round(float(age), 1) if age is not None else None,
                    "fps": round(max(fps, 0.0), 1),
                    "last_alert_at": last_alert,
                    "last_alert_type": getattr(manager, "last_alert_type", None),
                })
            previous_at = now
            atomic_write_json(self.path, {
                "pipeline": "running",
                "config_revision": self.config_revision,
                "started_at": self._started_at,
                "updated_at": now_iso(),
                "cameras": cameras,
            })
