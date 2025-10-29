FROM nvcr.io/nvidia/tritonserver:24.11-py3

WORKDIR /models

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.8.0+cu126 && \
    pip3 install --no-cache-dir \
        opencv-python \
        accelerate \
        transformers \
        torchvision \
        omegaconf \
        timm \
        easydict \
        scikit-image

COPY models_serving /models

RUN pip3 install --no-cache-dir /models/split_b/1/.cached_model/extractor/models/vit_kprpe/RPE/rpe_ops

EXPOSE 8000 8001

ENTRYPOINT ["tritonserver", "--model-repository=/models", "--log-verbose=0"]
