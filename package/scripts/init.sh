#!/usr/bin/env bash
# 首次使用：创建目录、复制配置模板、复制环境变量模板。
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p config models generated logs runtime

if [ ! -f config/config.yaml ]; then
    cp config/config.example.yaml config/config.yaml
    chmod 600 config/config.yaml
    echo "已创建 config/config.yaml（请按现场修改 RTSP/webhook）"
fi

if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    echo "已创建 .env（请修改 SAFETY_API_KEY）"
fi

echo "初始化完成。下一步："
echo "  1. 把主项目的 deploy/models/ 复制到 package/models/"
echo "  2. 修改 config/config.yaml 和 .env"
echo "  3. ./scripts/build.sh && ./scripts/start.sh"
