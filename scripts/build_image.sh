#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${1:-kfp-workflow:latest}"

echo "Building ${IMAGE_NAME} from ${PROJECT_ROOT}/docker/Dockerfile"
docker build -t "${IMAGE_NAME}" -f "${PROJECT_ROOT}/docker/Dockerfile" "${PROJECT_ROOT}"
echo "Done: ${IMAGE_NAME}"
