#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-multirocket-new:latest}"
docker build -t "${IMAGE_TAG}" .
