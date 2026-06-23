# Production Checklist — MKD Fine-Tuning Platform v1.0.0

Run before every production deployment.

## Security

- [x] No hardcoded secrets — `ADMIN_PASSWORD`, `HF_TOKEN`, `REVISION_SHA`, `ALLOWED_HOSTS` loaded from env
- [x] `_safe_name()` path-traversal guard on every file-access endpoint (api.py)
- [x] `TrustedHostMiddleware` rejects requests with unapproved `Host` headers
- [x] CORS restricted to `GET/POST/DELETE/OPTIONS` and explicit headers only
- [x] Pydantic `Field` constraints (max_length, ge/le, pattern) on all request models
- [x] `HF_TOKEN` never accepted in request bodies — server env only
- [x] `REVISION_SHA` pins HF Hub commit to prevent supply-chain substitution
- [x] Non-root `mkd` user in Dockerfile

## Correctness

- [x] All locks used properly (`_active_job_lock`, `_vllm_lock`, `_inf_lock`)
- [x] `_compute_vqa_accuracy` raises `RuntimeError` if >50 % of samples fail
- [x] `prepare_inputs_for_generation` patched; no `NoneType.shape` crash in VL inference
- [x] `is_ready()` returns `False` instead of raising when vLLM is not started
- [x] `parse_training_log` handles malformed ANSI sequences without crashing

## Code Quality

- [x] `platform/` renamed to `ft_platform/` — no stdlib module shadowing
- [x] `python -m py_compile app.py api.py` passes
- [x] pytest 177 tests pass, 81 % coverage (GPU/Gradio-only files omitted)
- [x] `pytest.ini` uses `--import-mode=importlib` to prevent path collisions
- [x] `.coveragerc` documents which files are intentionally omitted and why

## Architecture

- [x] Monolithic `app.py` refactored into `inference/`, `ft_platform/`, `ui/` packages
- [x] `api.py` is the single REST layer; Gradio UI and REST API are independent
- [x] CSS externalized to `ui/styles.py`
- [x] `ui/layout.py` contains the Gradio `gr.Blocks` definition
- [x] `inference/state.py` holds all global mutable state and locks

## Infrastructure

- [x] Dockerfile exists (multi-stage, NVIDIA CUDA 12.4.1, non-root user)
- [x] `docker-compose.yml` — api + ui + redis with GPU reservations
- [x] `Makefile` — `make test`, `make lint`, `make security`, `make docker`
- [x] `.github/workflows/ci.yml` — lint, security, test jobs
- [x] `pyproject.toml` (PEP 621) replaces `setup.py`

## Documentation

- [x] `README.md` — architecture, security model, env vars, migration guide
- [x] `API_SPEC.md` — all endpoints, error codes, field constraints
- [x] `FRONTEND_CONTRACT.md` — request/response shapes for UI integration
- [x] `CHANGELOG.md` — all S/B/R changes from original
- [x] Health check endpoint `GET /api/health`

## Pre-Launch Commands

```bash
cd /home/mkd-s002/fine-tune/ft-platform-refactor

# Syntax check
python -m py_compile app.py api.py inference/state.py inference/vllm_server.py \
    ft_platform/log_parser.py ft_platform/eval_dashboard.py ft_platform/training.py \
    ui/components.py ui/styles.py

# Full test suite with coverage
/home/mkd-s002/fine-tune/venv/bin/pytest tests/ --cov=. --cov-config=.coveragerc \
    --cov-report=term-missing -q

# Formatting check
black --check --line-length=100 api.py inference/ ft_platform/ ui/

# Linting
ruff check api.py inference/ ft_platform/ ui/

# Security scan
bandit -r api.py inference/ ft_platform/ ui/ -ll

# Docker build (requires NVIDIA Container Toolkit)
docker build -t mkd-finetuning-platform:v1.0.0 .
```

## Known Omissions (intentional, documented)

| File | Why omitted from test coverage |
|------|-------------------------------|
| `inference/chat.py` | Requires torch + CUDA GPU; all branches touch GPU state |
| `vl_trainer.py` | Training loop requires GPU memory; 700-line torch-heavy script |
| `ui/layout.py` | Requires Gradio to instantiate `gr.Blocks`; import fails without display |
| `app.py` | Entry-point bootstrap only; not a library module |
| `pdf_utils.py` | Standalone utility script, not imported by the platform |
| `sanity_check.py` | Standalone diagnostic script, not imported by the platform |
