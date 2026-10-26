"""控制面 API 的小工具：时间、RTSP 脱敏、原子写文件。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mask_rtsp_url(url: str | None) -> str | None:
    """把 rtsp://user:password@host 变成 rtsp://user:***@host。"""
    if not url:
        return url
    try:
        parsed = urlsplit(url)
        if parsed.password is None:
            return url
        host = parsed.hostname or ""
        netloc = f"{parsed.username or ''}:***@{host}"
        if parsed.port:
            netloc += f":{parsed.port}"
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
    except Exception:
        return url


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def atomic_write_json(path: str | Path, data) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
