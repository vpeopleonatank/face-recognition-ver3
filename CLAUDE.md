# Repository Guidelines

## Project Structure & Module Organization
`app/` hosts the FastAPI service: `main.py` wires up dependencies, `api/v1/routes.py` exposes REST endpoints, and `core/`, `services/`, and `utils/` hold settings, Triton/rerank clients, and shared helpers. Integration tests live in `tests/` and mirror public APIs. Model artifacts, CUDA extensions, and offline helpers reside in `triton-infer-custom/`, `face_v3/`, and `scripts/`; keep generated plans and shared objects inside those directories so the Docker build can copy them verbatim. Reference documents are under `docs/`.

## Build, Test, and Development Commands
Activate a virtualenv (`python -m venv .venv && source .venv/bin/activate`) and install dependencies with `pip install -r requirements.txt` plus `pip install -r requirements-dev.txt` for test tooling. Run the API locally via `uvicorn app.main:app --reload` once `FACE_V3_*` variables are set (copy `.env.example`). Execute unit and integration tests with `pytest`. Use `docker compose up --build` for an end-to-end stack that includes Triton, and `./triton-infer-custom/build.sh` when ONNX models change to regenerate TensorRT plans.

## Coding Style & Naming Conventions
Target Python 3.10 (see `Dockerfile`), four-space indentation, and follow PEP 8 plus the existing type-hinted patterns. Keep module-level constants uppercase, request/response schemas in `schemas/` named with `*Schema`, and log via `loguru` using structured messages. When adding utilities, colocate them with the closest domain module (`utils/image.py`, `services/triton_client.py`) and document non-obvious behavior with concise docstrings.

## Testing Guidelines
Prefer pure-Python unit tests that stub Triton and rerank integrations, mirroring the approach in `tests/test_api_endpoints.py`. Place new tests in `tests/` with filenames matching `test_*.py` and fixtures in `conftest.py`. Validate both happy-path and failure-path behavior, and ensure any new API contract updates include JSON assertions. Capture local verification with `pytest -vv` (add `-k` filters for focused runs) before opening a PR.

## Commit & Pull Request Guidelines
Commit subjects generally follow Conventional Commits (`feat:`, `fix:`) or concise imperative sentences; keep them under 72 characters and describe scope and intent. Each PR should include a summary of changes, configuration notes (e.g., new environment variables), and test evidence (`pytest`, compose logs). Link tracking issues and attach relevant screenshots or sample responses when API behavior shifts so reviewers can validate quickly.

## Security & Configuration Tips
Never commit real `.env` values or proprietary model binaries; use `.env.example` and document required keys instead. Treat shared libraries in `face_v3/libs/` and `triton-infer-custom/models_serving/` as deployable artifacts—update checksums or source locations when regenerating them. Review changes for GPU resource assumptions and note any new ports or host mounts introduced by Docker updates.
