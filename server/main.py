#!/usr/bin/env python3
"""
安全帽 / 反光衣 / 安全带检测服务端入口（DeepStream 两阶段：整帧检测 + 二级分类器）

启动方式:
    python tools/model_build.py --config deploy/config.yaml   # ① 编译模型产物（generated/，增量）
    python -m server.main --config deploy/config.yaml          # ② 启动服务
    （容器内由 SAFETY_CONFIG 指定配置；部署根 = config.yaml 所在目录）

架构:
    本模块 = 主进程监督循环（supervisor）
        └── server/service.py:serve  = 子进程：日志 + 看门狗 + DeepStream 管线
                └── server/pipeline/builder.py:PipelineBuilder = 管线构建

    RTSP 源×N → nvstreammux(batch=N) → [detector/classifier 链] → nvstreamdemux
    → 每路: tee → [ appsink(原始证据帧) | nvdsosd → shmsink(→RTSP, 仅开启 output) ]

断流自愈（Plan A + B）:
    A. nvurisrcbin 内置重连参数收敛为有限次数（source.reconnect.attempts），
       timeout/interval 提至 10s（5s 在 DS7.x 上易触发重连风暴后永久卡死，
       见 NVIDIA 论坛 nvurisrcbin reconnect 相关议题）。
    B. 子进程内 StreamWatchdog 监听 gst stdout：
       重连循环无进展 / 长时间无健康帧心跳 → 以 EXIT_PIPELINE_STUCK 退出；
       主进程监督循环退避重启、全量重建管线。无限重试(-1)已被证明会死锁，
       因此不再使用。
"""

import argparse
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from server.config import EXIT_PIPELINE_STUCK, load_config

# 兼容旧引用（tools/explore_metadata.py 等）从 server.main 导入的入口
from server.pipeline.ini_patch import anchor_ini_config as _anchor_ini_config  # noqa: F401


def _runtime_dir(deploy_root: Path) -> Path:
    return Path(os.environ.get("SAFETY_RUNTIME_DIR") or (deploy_root / "runtime"))


def _read_revision(runtime_dir: Path) -> int:
    try:
        import json
        return int(json.loads((runtime_dir / "revision.json").read_text(
            encoding="utf-8"))["revision"])
    except Exception:
        return 0


def _run_supervised(config_path: str, deploy_root: Path) -> int:
    """监督循环：启动子进程；配置变更时优雅重启；异常退出按退避策略重建。"""
    runtime_dir = _runtime_dir(deploy_root)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    reload_file = runtime_dir / "reload.request"
    ready_file = runtime_dir / "ready"
    last_reload_ns = reload_file.stat().st_mtime_ns if reload_file.exists() else 0
    attempt = 0

    while True:
        attempt += 1
        try:
            config = load_config(config_path)
        except Exception as exc:
            print(f"配置加载失败，监督循环退出: {exc}", flush=True)
            return 1

        revision = _read_revision(runtime_dir)
        os.environ["SAFETY_CONFIG_REVISION"] = str(revision)
        try:
            ready_file.unlink()
        except FileNotFoundError:
            pass

        proc = Process(target=_child_target, args=(config, deploy_root),
                       name=f"safety-pipeline-{attempt}")
        proc.start()
        reload_requested = False
        try:
            while True:
                proc.join(timeout=1)
                if proc.exitcode is not None:
                    break
                if reload_file.exists() and reload_file.stat().st_mtime_ns > last_reload_ns:
                    last_reload_ns = reload_file.stat().st_mtime_ns
                    reload_requested = True
                    print(f"收到配置重载请求 (revision={_read_revision(runtime_dir)})，"
                          f"重启管线 process={proc.name}", flush=True)
                    proc.terminate()
                    proc.join(10)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                    break
        except KeyboardInterrupt:
            print("\nInterrupted. Terminating...")
            proc.terminate()
            proc.join()
            return 0

        if reload_requested:
            attempt = 0
            continue

        exitcode = proc.exitcode
        if exitcode is None or exitcode == 0:
            return exitcode or 0

        wait = min(2 * attempt, 30)
        # 生命周期锚点：process 与子进程内 fatal 的 process 同源
        #（子进程的 multiprocessing.current_process().name）；老关键字保留做兼容。
        print(f"pipeline process={proc.name} exited code={exitcode} (attempt {attempt}), "
              f"restarting in {wait}s", flush=True)
        time.sleep(wait)


def _child_target(config: dict, deploy_root: Path) -> None:
    from server.service import serve
    serve(config, deploy_root)


def main():
    parser = argparse.ArgumentParser(description="安全帽/反光衣检测服务端 (DeepStream)")
    parser.add_argument(
        "--config", "-c",
        default=os.environ.get("SAFETY_CONFIG")
                or str(_PROJECT_ROOT / "deploy" / "config.yaml"),
        help="配置文件路径（默认: $SAFETY_CONFIG 或 <项目根>/deploy/config.yaml）",
    )
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    print(f"加载配置: {config_path}")
    load_config(config_path)  # 启动前 fail fast；监督循环会按需重新加载
    deploy_root = Path(config_path).parent
    return _run_supervised(config_path, deploy_root)


if __name__ == "__main__":
    sys.exit(main())
