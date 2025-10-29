FROM nvcr.io/nvidia/tritonserver:24.11-py3

# Set working directory
WORKDIR /models

# Install dependencies
RUN apt-get update && apt-get install -y \
      build-essential \
      ninja-build \
      libgl1 \
      libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.8.0+cu126 && \
    pip3 install --no-cache-dir opencv-python accelerate transformers torchvision omegaconf timm easydict scikit-image

COPY models_serving /models

# Expose gRPC port
EXPOSE 8001

ENTRYPOINT ["sh", "-c", "pip3 install /models/split_b/1/.cached_model/extractor/models/vit_kprpe/RPE/rpe_ops && tritonserver --model-repository=/models"]
