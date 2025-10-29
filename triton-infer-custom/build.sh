#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build TensorRT engines and the custom Triton image.

Usage: ./build.sh [--no-run] [--tag <image_name>] [--workspace <path>]

Options:
  --no-run           Skip starting the Triton container after the build.
  --tag <image>      Override the resulting Triton image tag (default: tritonserver-plan-vnd:24.11-py3).
  --workspace <dir>  Path mounted into the TensorRT container (default: current directory).
  -h, --help         Show this help message and exit.
EOF
}

RUN_CONTAINER=1
TRITON_CUSTOM_IMAGE="tritonserver-plan-vnd:24.11-py3"
WORKSPACE="$(pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-run)
      RUN_CONTAINER=0
      shift
      ;;
    --tag)
      TRITON_CUSTOM_IMAGE="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

MODEL_REPO_HOST="${WORKSPACE}/triton-infer-custom/models_serving"
MODEL_REPO_CONTAINER="/workspace/triton-infer-custom/models_serving"
MODEL1_NAME="scrfd"
MODEL2_NAME="split_a"
ONNX1_PATH="${MODEL_REPO_HOST}/${MODEL1_NAME}/1/model.onnx"
ONNX1_CONTAINER_PATH="${MODEL_REPO_CONTAINER}/${MODEL1_NAME}/1/model.onnx"
ENGINE1_PATH="${MODEL_REPO_HOST}/${MODEL1_NAME}/1/model.plan"
ENGINE1_CONTAINER_PATH="${MODEL_REPO_CONTAINER}/${MODEL1_NAME}/1/model.plan"
ONNX2_PATH="${MODEL_REPO_HOST}/${MODEL2_NAME}/1/model.onnx"
ONNX2_CONTAINER_PATH="${MODEL_REPO_CONTAINER}/${MODEL2_NAME}/1/model.onnx"
ENGINE2_PATH="${MODEL_REPO_HOST}/${MODEL2_NAME}/1/model.plan"
ENGINE2_CONTAINER_PATH="${MODEL_REPO_CONTAINER}/${MODEL2_NAME}/1/model.plan"

TENSORRT_IMAGE="nvcr.io/nvidia/tensorrt:24.11-py3"

DOCKER_BIN=${DOCKER_BIN:-docker}

if ! command -v "${DOCKER_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] docker binary not found (looked for '${DOCKER_BIN}')." >&2
  exit 1
fi

if [[ ! -f "${ONNX1_PATH}" || ! -f "${ONNX2_PATH}" ]]; then
  echo "[ERROR] ONNX models not found. Expected at:" >&2
  echo "        ${ONNX1_PATH}" >&2
  echo "        ${ONNX2_PATH}" >&2
  exit 1
fi

echo "[INFO] Converting ONNX -> TensorRT engines using ${TENSORRT_IMAGE}"

"${DOCKER_BIN}" run --rm --gpus all \
  -v "${WORKSPACE}":/workspace \
  -w /workspace \
  "${TENSORRT_IMAGE}" trtexec \
    --onnx="${ONNX1_CONTAINER_PATH}" \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:8x3x640x640 \
    --maxShapes=input:32x3x640x640 \
    --saveEngine="${ENGINE1_CONTAINER_PATH}"

"${DOCKER_BIN}" run --rm --gpus all \
  -v "${WORKSPACE}":/workspace \
  -w /workspace \
  "${TENSORRT_IMAGE}" trtexec \
    --onnx="${ONNX2_CONTAINER_PATH}" \
    --minShapes=input:1x3x112x112 \
    --optShapes=input:8x3x112x112 \
    --maxShapes=input:32x3x112x112 \
    --saveEngine="${ENGINE2_CONTAINER_PATH}"

echo "[INFO] Engines saved to" "$ENGINE1_PATH" "and" "$ENGINE2_PATH"

echo "[INFO] Building Triton image ${TRITON_CUSTOM_IMAGE}"
"${DOCKER_BIN}" build -t "${TRITON_CUSTOM_IMAGE}" triton-infer-custom

if [[ ${RUN_CONTAINER} -eq 1 ]]; then
  echo "[INFO] Starting Triton container ${TRITON_CUSTOM_IMAGE}"
  "${DOCKER_BIN}" run --shm-size=1g --gpus all --network host -d "${TRITON_CUSTOM_IMAGE}"
else
  echo "[INFO] Skipping container startup (--no-run)"
fi
