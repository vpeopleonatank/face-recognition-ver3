FROM nvcr.io/nvidia/tritonserver:24.11-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /models

# Runtime libraries required by OpenCV and model code.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies. Keep torch/cu126 and model requirements only.
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.8.0+cu126 && \
    pip3 install --no-cache-dir \
        opencv-python-headless \
        transformers \
        torchvision \
        omegaconf \
        timm \
        accelerate \
        easydict \
        scikit-image

COPY models_serving /models

# Build custom CUDA extension and clean up build toolchain and artifacts.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000 8001

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
