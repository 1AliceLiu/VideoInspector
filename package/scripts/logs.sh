#!/usr/bin/env bash
# 查看检测容器和 API 容器日志。
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose logs -f --tail=200
