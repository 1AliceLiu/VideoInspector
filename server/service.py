#!/usr/bin/env python3
"""子进程入口：初始化日志 → 构建管线 → 运行。

正常情况下永不返回（p.wait() 阻塞）。异常/看门狗触发时以非零码退出，
由 server/main.py 的监督循环决定是否重建。
"""

import multiprocessing
import os
from pathlib import Path

from server.pipeline.builder import PipelineBuilder
from server.pipeline.health import SourceHealthMonitor
from server.utils.logger import setup_logger
from server.utils.stdio import forward_stdout_to_logger
from server.watchdog import StreamWatchdog


def serve(config: dict, deploy_root: Path) -> None:
    os.environ.setdefault("GST_DEBUG", "0")

    log_cfg = config.get("log") or {}
    log_file = os.environ.get("SAFETY_LOG_FILE") or log_cfg.get("file")
    if log_file and not Path(log_file).is_absolute():
        log_file = str(deploy_root / log_file)
    logger = setup_logger(
        name="safety_server",
        level=log_cfg.get("level", "INFO"),
        log_file=log_file,
    )
    src_cfg = config.get("source") or {}
    watchdog = StreamWatchdog(src_cfg.get("reconnect"),
                              maintenance_window=(src_cfg.get("health") or {}).get("maintenance_window"))
    forward_stdout_to_logger(logger, on_line=watchdog.feed)
    logger.info("startup safety_detection_server")

    builder = PipelineBuilder(config, deploy_root)
    health = SourceHealthMonitor(builder.cameras,
                                 (config.get("source") or {}).get("health"))
    runtime = builder.build(logger, health=health)
    health.start()

    runtime_dir = os.environ.get("SAFETY_RUNTIME_DIR")
    revision = 0
    status_writer = None
    ready_path = None
    if runtime_dir:
        from server.api.status_writer import StatusWriter

        runtime_path = Path(runtime_dir)
        runtime_path.mkdir(parents=True, exist_ok=True)
        try:
            revision = int(os.environ.get("SAFETY_CONFIG_REVISION", "0"))
        except ValueError:
            revision = 0
        status_writer = StatusWriter(
            runtime_path, builder.cameras, health,
            managers=getattr(builder, "alert_managers", None),
            config_revision=revision,
        )
        ready_path = runtime_path / "ready"

    # 生命周期锚点：与 fatal/restart 行的 process 同源，供重启因果链关联新一代进程。
    logger.info("pipeline started process={} cameras={}",
                multiprocessing.current_process().name, len(builder.cameras))
    pipeline = runtime["pipeline"]
    try:
        running = pipeline.start()
        if status_writer is not None:
            status_writer.start()
        if ready_path is not None:
            ready_path.write_text(str(revision), encoding="utf-8")
        (running if hasattr(running, "wait") else pipeline).wait()
    finally:
        if ready_path is not None:
            try:
                ready_path.unlink()
            except OSError:
                pass
        if status_writer is not None:
            status_writer.stop()
        if runtime_dir:
            try:
                import json
                from server.api.util import atomic_write_json, now_iso
                status_path = Path(runtime_dir) / "status.json"
                try:
                    data = json.loads(status_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
                data["pipeline"] = "stopped"
                data["config_revision"] = revision
                data["updated_at"] = now_iso()
                for cam in data.get("cameras") or []:
                    cam["online"] = False
                    cam["last_frame_age_seconds"] = None
                    cam["fps"] = 0
                atomic_write_json(status_path, data)
            except Exception:
                pass
        health.stop()
        if runtime["rtsp_server"] is not None:
            runtime["rtsp_server"].stop()
        runtime["executor"].shutdown(wait=False, cancel_futures=True)
