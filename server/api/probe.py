"""RTSP 探测：优先 ffprobe，缺失时退回 RTSP OPTIONS。

只用于控制面排障，不参与检测热路径。结果里的 URL 一律脱敏。
"""

from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import time
from urllib.parse import urlsplit, urlunsplit

from server.api.util import mask_rtsp_url


def _fps(value) -> float | None:
    try:
        if not value:
            return None
        if "/" in str(value):
            num, den = str(value).split("/", 1)
            den_f = float(den)
            return round(float(num) / den_f, 2) if den_f else None
        return round(float(value), 2)
    except Exception:
        return None


def _probe_ffprobe(url: str, timeout: float) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-rtsp_transport", "tcp",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate",
        "-of", "json", url,
    ]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    latency = round((time.monotonic() - started) * 1000)
    if proc.returncode != 0:
        return {
            "reachable": False,
            "source_mode": None,
            "resolved_rtsp_url": mask_rtsp_url(url),
            "resolution": None,
            "codec": None,
            "fps": None,
            "latency_ms": latency,
            "error": proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "ffprobe failed",
        }
    streams = (json.loads(proc.stdout or "{}").get("streams") or [])
    stream = streams[0] if streams else {}
    width, height = stream.get("width"), stream.get("height")
    return {
        "reachable": True,
        "source_mode": None,
        "resolved_rtsp_url": mask_rtsp_url(url),
        "resolution": f"{width}x{height}" if width and height else None,
        "codec": stream.get("codec_name"),
        "fps": _fps(stream.get("avg_frame_rate")),
        "latency_ms": latency,
        "error": None,
    }


def _probe_options(url: str, timeout: float) -> dict:
    """最小 RTSP OPTIONS 探测：只验证端口和 RTSP 应答，不解析 SDP。"""
    parsed = urlsplit(url)
    host = parsed.hostname
    port = parsed.port or 554
    if not host:
        return {
            "reachable": False, "source_mode": None,
            "resolved_rtsp_url": mask_rtsp_url(url),
            "resolution": None, "codec": None, "fps": None,
            "latency_ms": None, "error": "RTSP URL 缺少 host",
        }

    request_url = urlunsplit((parsed.scheme, parsed.netloc.split("@")[-1], parsed.path or "/", parsed.query, ""))
    headers = [
        f"OPTIONS {request_url} RTSP/1.0",
        "CSeq: 1",
        "User-Agent: VideoInspector/0.2",
    ]
    if parsed.username is not None:
        credentials = f"{parsed.username}:{parsed.password or ''}".encode("utf-8")
        headers.append("Authorization: Basic " + base64.b64encode(credentials).decode("ascii"))
    request = "\r\n".join(headers) + "\r\n\r\n"

    started = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request.encode("utf-8"))
            data = sock.recv(4096)
        latency = round((time.monotonic() - started) * 1000)
        text = data.decode("utf-8", errors="replace")
        reachable = "200" in text.splitlines()[0] if text else False
        return {
            "reachable": reachable,
            "source_mode": None,
            "resolved_rtsp_url": mask_rtsp_url(url),
            "resolution": None,
            "codec": None,
            "fps": None,
            "latency_ms": latency,
            "error": None if reachable else (text.splitlines()[0] if text else "empty response"),
        }
    except Exception as exc:
        return {
            "reachable": False, "source_mode": None,
            "resolved_rtsp_url": mask_rtsp_url(url),
            "resolution": None, "codec": None, "fps": None,
            "latency_ms": None, "error": str(exc),
        }


def probe_rtsp(url: str | None, timeout: float = 10) -> dict:
    if not url:
        return {
            "reachable": False, "source_mode": None,
            "resolved_rtsp_url": None,
            "resolution": None, "codec": None, "fps": None,
            "latency_ms": None, "error": "rtsp_url 为空",
        }
    if shutil.which("ffprobe"):
        try:
            return _probe_ffprobe(url, timeout)
        except subprocess.TimeoutExpired:
            return {
                "reachable": False, "source_mode": None,
                "resolved_rtsp_url": mask_rtsp_url(url),
                "resolution": None, "codec": None, "fps": None,
                "latency_ms": round(timeout * 1000), "error": "ffprobe 超时",
            }
        except Exception:
            pass
    return _probe_options(url, timeout)
