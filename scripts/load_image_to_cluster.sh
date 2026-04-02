#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="${1:-kfp-workflow:latest}"
TARGET_NODE="${2:-scouter1}"
NAMESPACE="${IMAGE_LOADER_NAMESPACE:-image-loader}"
HELPER_IMAGE="${IMAGE_LOADER_HELPER_IMAGE:-ghcr.io/containerd/nerdctl:v2.1.6}"

sanitize() {
  printf '%s' "$1" | tr '/:' '__'
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

require_cmd docker
require_cmd kubectl

IMAGE_SLUG="$(sanitize "${IMAGE_NAME}")"
STAMP="$(date +%Y%m%d%H%M%S)"
ARCHIVE_PATH="/tmp/${IMAGE_SLUG}_${STAMP}.tar"
POD_NAME="image-importer-${STAMP}"

cleanup() {
  kubectl delete pod "${POD_NAME}" -n "${NAMESPACE}" --ignore-not-found >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Saving ${IMAGE_NAME} to ${ARCHIVE_PATH}"
docker save "${IMAGE_NAME}" -o "${ARCHIVE_PATH}"

echo "Ensuring namespace ${NAMESPACE} exists with privileged pod security labels"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace "${NAMESPACE}" \
  pod-security.kubernetes.io/enforce=privileged \
  pod-security.kubernetes.io/audit=privileged \
  pod-security.kubernetes.io/warn=privileged \
  --overwrite >/dev/null

echo "Creating importer pod ${POD_NAME} on node ${TARGET_NODE}"
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
spec:
  restartPolicy: Never
  nodeName: ${TARGET_NODE}
  containers:
    - name: importer
      image: ${HELPER_IMAGE}
      imagePullPolicy: IfNotPresent
      command:
        - sh
        - -lc
        - nerdctl --address /run/containerd/containerd.sock --namespace k8s.io load -i /host-tmp/$(basename "${ARCHIVE_PATH}")
      securityContext:
        privileged: true
      volumeMounts:
        - name: containerd
          mountPath: /run/containerd/containerd.sock
        - name: host-tmp
          mountPath: /host-tmp
  volumes:
    - name: containerd
      hostPath:
        path: /run/containerd/containerd.sock
        type: Socket
    - name: host-tmp
      hostPath:
        path: /tmp
        type: Directory
EOF

echo "Waiting for importer pod completion"
kubectl wait --for=condition=Ready=False "pod/${POD_NAME}" -n "${NAMESPACE}" --timeout=10s >/dev/null 2>&1 || true
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded "pod/${POD_NAME}" -n "${NAMESPACE}" --timeout=180s >/dev/null

echo "Importer logs"
kubectl logs "${POD_NAME}" -n "${NAMESPACE}"

echo "Image ${IMAGE_NAME} imported to node ${TARGET_NODE}"
