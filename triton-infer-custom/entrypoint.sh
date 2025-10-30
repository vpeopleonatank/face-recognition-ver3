#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=python3

check_module() {
  ${PYTHON_BIN} - <<'PY'
import importlib
mod = importlib.import_module("rpe_index_cpp")
if not hasattr(mod, "forward_gpu"):
    raise AttributeError("forward_gpu missing")
PY
}

if ! check_module >/dev/null 2>&1; then
  echo "[INFO] Installing rpe_ops with CUDA kernels..."
  ${PYTHON_BIN} -m pip install --no-cache-dir /models/split_b/1/.cached_model/extractor/models/vit_kprpe/RPE/rpe_ops
  check_module >/dev/null
else
  echo "[INFO] rpe_index_cpp already has forward_gpu, skipping install"
fi

exec tritonserver --model-repository=/models --log-verbose=0
