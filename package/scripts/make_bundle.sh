#!/usr/bin/env bash
# 生成可发给同事的交付包（默认不含模型，模型体积太大）：
#   ./scripts/make_bundle.sh
# 连模型一起打包（体积可能很大）：
#   WITH_MODELS=1 ./scripts/make_bundle.sh
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
VERSION="${VERSION:-v0.2}"
BUNDLE_NAME="video-inspector-api-${VERSION}"
DIST_DIR="${PACKAGE_DIR}/dist"
BUNDLE="${DIST_DIR}/${BUNDLE_NAME}"

rm -rf "${BUNDLE}" "${DIST_DIR}/${BUNDLE_NAME}.tar.gz" "${DIST_DIR}/${BUNDLE_NAME}.tar.gz.sha256"
mkdir -p "${BUNDLE}/package/models" "${BUNDLE}/package/generated" \
         "${BUNDLE}/package/config" "${BUNDLE}/package/logs" "${BUNDLE}/package/runtime"

# 代码与构建文件（不要带真实 config.yaml / 模型 / 日志 / 引擎）
cp -a "${REPO_ROOT}/server" "${BUNDLE}/"
cp -a "${REPO_ROOT}/tools" "${BUNDLE}/"
cp -a "${REPO_ROOT}/deploy" "${BUNDLE}/"
cp -a "${REPO_ROOT}/requirements.txt" "${BUNDLE}/"
cp -a "${REPO_ROOT}/API-v1.md" "${BUNDLE}/"
cp -a "${REPO_ROOT}/.dockerignore" "${BUNDLE}/"
rm -rf "${BUNDLE}/deploy/models" "${BUNDLE}/deploy/generated" \
       "${BUNDLE}/deploy/logs" "${BUNDLE}/deploy/alerts" \
       "${BUNDLE}/deploy/config.yaml" "${BUNDLE}/deploy/README.md" \
       "${BUNDLE}/deploy/docker-compose.yml"
find "${BUNDLE}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${BUNDLE}" -type f -name "*.pyc" -delete

# 交付包自身
cp -a "${PACKAGE_DIR}/docker-compose.yml" "${BUNDLE}/package/"
cp -a "${PACKAGE_DIR}/.env.example" "${BUNDLE}/package/"
cp -a "${PACKAGE_DIR}/config/config.example.yaml" "${BUNDLE}/package/config/"
cp -a "${PACKAGE_DIR}/docs" "${BUNDLE}/package/"
cp -a "${PACKAGE_DIR}/scripts" "${BUNDLE}/package/"
cp -a "${PACKAGE_DIR}/models/README.md" "${BUNDLE}/package/models/"
cp -a "${PACKAGE_DIR}/README.md" "${BUNDLE}/README.md"
cp -a "${PACKAGE_DIR}/README.md" "${BUNDLE}/package/README.md"
touch "${BUNDLE}/package/generated/.gitkeep" \
      "${BUNDLE}/package/logs/.gitkeep" \
      "${BUNDLE}/package/runtime/.gitkeep"

if [ "${WITH_MODELS:-0}" = "1" ]; then
    echo "正在复制模型（可能很慢）..."
    cp -a "${REPO_ROOT}/deploy/models/." "${BUNDLE}/package/models/"
fi

mkdir -p "${DIST_DIR}"
tar -C "${DIST_DIR}" -czf "${DIST_DIR}/${BUNDLE_NAME}.tar.gz" "${BUNDLE_NAME}"
sha256sum "${DIST_DIR}/${BUNDLE_NAME}.tar.gz" \
    | sed "s#${DIST_DIR}/##" > "${DIST_DIR}/${BUNDLE_NAME}.tar.gz.sha256"

echo
echo "交付包已生成："
echo "  ${DIST_DIR}/${BUNDLE_NAME}.tar.gz"
echo "  ${DIST_DIR}/${BUNDLE_NAME}.tar.gz.sha256"
echo
echo "包含：代码 + Dockerfile + package 配置/脚本/文档"
if [ "${WITH_MODELS:-0}" != "1" ]; then
    echo "未包含模型：请把 deploy/models/ 复制到解压后的 package/models/ 再启动。"
fi
