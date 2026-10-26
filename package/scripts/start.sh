#!/usr/bin/env bash
# 后台启动检测容器 + API 容器。
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose up -d
docker compose ps
