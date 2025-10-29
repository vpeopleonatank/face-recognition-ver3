#!/bin/bash
set -e

# ========================
# Config
# ========================
MODEL_REPO="./models_serving"
MODEL1_NAME="scrfd"
MODEL2_NAME="split_a"
ONNX1_PATH="${MODEL_REPO}/${MODEL1_NAME}/1/model.onnx"
ENGINE1_PATH="${MODEL_REPO}/${MODEL1_NAME}/1/model.plan"
ONNX2_PATH="${MODEL_REPO}/${MODEL2_NAME}/1/model.onnx"
ENGINE2_PATH="${MODEL_REPO}/${MODEL2_NAME}/1/model.plan"

TENSORRT_IMAGE="nvcr.io/nvidia/tensorrt:24.11-py3"
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:24.11-py3"

TRITON_CUSTOM_IMAGE="tritonserver-plan-vnd:24.11-py3"

# ========================
# Step 1: Convert ONNX -> TensorRT engine
# ========================
echo "[INFO] Converting ONNX -> TensorRT engine..."

sudo docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    $TENSORRT_IMAGE trtexec \
      --onnx=$ONNX1_PATH \
      --minShapes=input:1x3x640x640 \
      --optShapes=input:8x3x640x640 \
      --maxShapes=input:32x3x640x640 \
      --saveEngine=$ENGINE1_PATH

sudo docker run --rm --gpus all \
    -v $(pwd):/workspace \
    -w /workspace \
    $TENSORRT_IMAGE trtexec \
      --onnx=$ONNX2_PATH \
      --minShapes=input:1x3x112x112 \
      --optShapes=input:8x3x112x112 \
      --maxShapes=input:32x3x112x112 \
      --saveEngine=$ENGINE2_PATH

echo "[INFO] Engine saved to $ENGINE2_PATH"

# ========================
# Step 2: Build Triton Inference Server
# ========================
echo "[INFO] Build image triton..."
sudo docker build -t $TRITON_CUSTOM_IMAGE .

# ========================
# Step 3: Run Triton Inference Server
# ========================
echo "[INFO] Run image triton..."
sudo docker run --shm-size=1g --gpus all --network host -d $TRITON_CUSTOM_IMAGE
