# Model v3 Backend – Implementation Task Breakdown

## Phase 1 – Project Skeleton & Dependencies
1. Scaffold FastAPI project structure (`app/main.py`, `app/api/v1/routes.py`, `app/services`, `app/core`, `app/schemas`, `app/utils`).
2. Define Python requirements (FastAPI, uvicorn, tritonclient[gRPC], numpy, opencv-python, loguru, pydantic-settings).
3. Add configuration module (`app/core/config.py`) with environment-driven settings for Triton URL, batch sizes, rerank threshold, optional flags.
4. Implement application factory in `app/main.py` with startup/shutdown handlers and dependency injection placeholders.

## Phase 2 – Triton Client & Image Utilities
5. Extract and adapt image preprocessing/alignment helpers from `face_v3/infer.py` into `app/utils/image.py`.
6. Implement `app/services/triton_client.py`:
   - Initialize gRPC `InferenceServerClient`.
   - Provide batched detection (`run_detection`) and extraction (`run_extraction`) helpers.
   - Expose higher-level methods (e.g., `detect_and_embed(images)`) handling chunking and post-processing.
7. Add error handling and logging wrappers; ensure model metadata is cached at startup.

## Phase 3 – Embeddings Endpoint
8. ✅ Define Pydantic request/response schemas for `/embeddings` (support multiple images, optional flags).
9. ✅ Implement endpoint handler in `app/api/v1/routes.py`:
   - Decode incoming images.
   - Invoke Triton client to get detections + embeddings.
   - Build response with bounding boxes, landmarks, confidences, embeddings, optional aligned faces.
10. ✅ Add validation for batch size, image dimensions, and aggregate timing metrics.

## Phase 4 – Rerank Service & Endpoint
11. ✅ Port the `RerankComputeCpp` wrapper from `face_v3/test_rerank.py` into `app/services/rerank.py`, adjusting paths and startup initialization.
12. ✅ Wire rerank singleton and expose dependency (`get_rerank_service`) for injection.
13. ✅ Create Pydantic schemas for `/rerank` request (query embedding, candidate embeddings/IDs, optional threshold) and response (scores, metadata).
14. ✅ Implement `/rerank` handler to validate embeddings, call the C++ wrapper, and return scores with latency diagnostics.

## Phase 5 – Health Checks & Instrumentation
15. ✅ Add `/healthz` endpoint verifying Triton connectivity and rerank library availability (optionally run a lightweight inference).
16. ✅ Integrate logging (loguru) and request-level tracing for inference latency, rerank latency, and batch statistics.
17. Add Prometheus-compatible metrics or simple counters if needed by ops (toggle via config).

## Phase 6 – Testing
18. ✅ Write unit tests for image utilities and rerank initialization (using pytest). *(tests/test_image_utils.py, tests/test_rerank_service.py)*
19. ✅ Mock Triton client to test `/embeddings` endpoint responses and error scenarios. *(tests/test_api_endpoints.py)*
20. ✅ Create deterministic rerank tests comparing expected and actual score outputs. *(tests/test_rerank_service.py)*
21. ✅ Prepare load/benchmark script to measure throughput with sample images (`face_v3/images_test`). *(scripts/load_test_embeddings.py; dev deps in requirements-dev.txt)*

## Phase 7 – Packaging & Deployment
22. Write Dockerfile for the FastAPI service:
    - Install system libs and Python deps.
    - Copy app code and shared libraries.
    - Set entrypoint to `uvicorn`.
23. Finalize Triton packaging:
    - Generate TensorRT plans by running `triton-infer-custom/build.sh` (ensures `model.plan` artifacts exist under `models_serving`).
    - Review `triton-infer-custom/Dockerfile` and update dependencies/entrypoint if additional custom ops or models are required.
    - Tag the resulting image (`tritonserver-plan-vnd:24.11-py3` by default) for docker-compose consumption.
24. Add configuration examples (`.env.example`, docker-compose snippet, or Helm values) referencing both the API and custom Triton services.
25. Document deployment steps in README or ops doc (environment variables, ports, health checks, Triton image build prerequisites).

## Phase 8 – Integration & Rollout
26. Run end-to-end tests in staging with the docker-compose stack (FastAPI + custom Triton) and sample images; verify embeddings and rerank outputs.
27. Coordinate with web backend team to consume `/embeddings` and `/rerank` endpoints (update API contract docs).
28. Plan production rollout: deploy service, update backend configuration to call new endpoints, monitor metrics/logs, and iterate on thresholds as needed.
