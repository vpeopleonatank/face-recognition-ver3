# Face Search Backend (Model v3)

FastAPI service exposing face detection, embedding extraction, and rerank scoring while delegating heavy inference to a custom Triton server. This README captures the packaging and deployment flow described in `docs/ver3_backend_plan.md`.

## Prerequisites
- Docker 24+
- NVIDIA Container Toolkit with at least one CUDA-capable GPU
- Access to `nvcr.io` images used by TensorRT and Triton (requires NVIDIA NGC account)
- Generated TensorRT plans for the detection and embedding models

## Build The Triton Image
1. Ensure the ONNX models live under `triton-infer-custom/models_serving/<model>/1/model.onnx`.
2. Run the helper script to convert ONNX → TensorRT engines and bake the Triton image:
   ```bash
   ./triton-infer-custom/build.sh
   ```
   - Use `./triton-infer-custom/build.sh --no-run` to skip starting the container after build.
   - Override the resulting image tag with `--tag my-triton-image:latest` if desired.
   - The script relies on `nvcr.io/nvidia/tensorrt:24.11-py3`; authenticate with NGC first, e.g. `docker login nvcr.io`.

The script produces `model.plan` artifacts inside `triton-infer-custom/models_serving/**/1/` and builds the Docker image `tritonserver-plan-vnd:24.11-py3`, which the compose stack references.

## Build The FastAPI Image
1. Copy `.env.example` to `.env` and adjust values, especially:
   - `FACE_V3_API_KEY=` (set to the shared secret required in the `X-API-KEY` header)
   - `FACE_V3_TRITON_URL=triton:8001`
   - `FACE_V3_RERANK_LIBRARY_PATH=/opt/face_backend/libs/librerank_compute.so`
   - Batch sizes, model names, or thresholds as needed.
2. Build the API image (optional; docker compose will build on demand):
   ```bash
   docker build -t face-search-backend:latest .
   ```

The Dockerfile bundles the FastAPI app, shared libraries (`librerank_compute.so`, `libmerge_embeddings.so`), and runtime dependencies on top of a CUDA runtime image.

## Run Locally With Docker Compose
```bash
docker compose up --build
```

Services:
- `api` (port 8000) – FastAPI server exposing `/api/v1/embeddings`, `/api/v1/rerank`, and `/healthz`.
- `triton` (port 8001) – Custom Triton server loaded with the v3 detection and embedding ensembles. On first start it compiles the CUDA `rpe_ops` extension inside the container (uses the GPU); this step is cached on subsequent runs.

The API waits for Triton to pass its readiness probe (`http://triton:8000/v2/health/ready`) before starting.

## Health Checks & Observability
- FastAPI: `GET http://localhost:8000/healthz` validates Triton connectivity and rerank library availability.
- Triton: Compose health check polls the HTTP readiness endpoint exposed by Triton on port 8000 inside the container.

## Environment Reference (`.env`)
| Variable | Description | Default |
| --- | --- | --- |
| `FACE_V3_ENV` | Deployment environment flag controlling debug docs | `local` |
| `FACE_V3_TRITON_URL` | gRPC endpoint for Triton | `triton:8001` |
| `FACE_V3_DETECTION_MODEL_NAME` | Triton model used for detection | `detection` |
| `FACE_V3_EXTRACTION_MODEL_NAME` | Triton embedding model | `extraction` |
| `FACE_V3_DETECTION_INPUT_WIDTH/HEIGHT` | Detection input resolution | `640` |
| `FACE_V3_MAX_BATCH_SIZE` | Max images per batch | `8` |
| `FACE_V3_API_KEY` | Shared secret required in the `X-API-KEY` header | _(empty)_ |
| `FACE_V3_RERANK_LIBRARY_PATH` | Path to `librerank_compute.so` inside the API container | `/opt/face_backend/libs/librerank_compute.so` |
| `FACE_V3_RERANK_THRESHOLD` | Default rerank acceptance threshold | `0.5` |
| `FACE_V3_RETURN_ALIGNED` | Include aligned crops in responses | `false` |
| `FACE_V3_SKIP_EMBEDDING_NORMALIZATION` | Disable embedding L2 normalization | `false` |
| `UVICORN_WORKERS` | Uvicorn worker count injected via Docker CMD | `1` |

## Cleanup
Stop and remove containers and networks created by compose:
```bash
docker compose down
```

Regenerate TensorRT engines by rerunning `build.sh` whenever the ONNX models change.

If the Triton image is rebuilt, run `docker compose build triton` (or `docker compose up --build`) to refresh the container so the startup hook can rebuild the CUDA extension.
