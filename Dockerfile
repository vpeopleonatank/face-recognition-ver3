FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/opt/face_backend" \
    UVICORN_WORKERS=1

# Install Python runtime and system dependencies required by OpenCV and gRPC.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    python3.10 -m venv /opt/venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/face_backend

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r requirements.txt

COPY app ./app
COPY face_v3 ./face_v3

# Bundle shared libraries used by the rerank service and post-processing steps.
COPY face_v3/libs/librerank_compute.so ./libs/
COPY triton-infer-custom/models_serving/postprocess_extraction/1/libmerge_embeddings.so ./libs/

ENV FACE_V3_RERANK_LIBRARY_PATH="/opt/face_backend/libs/librerank_compute.so"

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-1}"]
