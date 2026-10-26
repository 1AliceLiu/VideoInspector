#!/usr/bin/env bash
# 构建检测镜像和控制面 API 镜像；首次会拉取 DeepStream 基础镜像，耗时较长。
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build
