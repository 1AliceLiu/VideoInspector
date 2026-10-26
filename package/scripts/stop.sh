#!/usr/bin/env bash
# 停止并移除容器（保留 config/models/generated/logs/runtime 数据）。
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose down
