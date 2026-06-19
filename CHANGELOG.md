# Changelog

All notable changes from the original monolithic `app.py` to the production-ready `v1.0.0` release.

## [1.0.0] — 2026-06-19

### Security Fixes

| ID | Change |
|----|--------|
| S1 | `_safe_name()` path-traversal guard added to every file-access endpoint |
| S2 | `ADMIN_PASSWORD` replaces hardcoded `DEV_PASSWORD`; loaded from env, not source |
| S3 | `HF_TOKEN` removed from all API request bodies; read from server env only |
| S4 | `PublishRequest.hf_token` field removed; `HF_TOKEN` env var is the only path |
| S5 | `REVISION_SHA` env var added; passed as `revision=` to every `from_pretrained` call to prevent supply-chain substitution |
| S6 | `TrustedHostMiddleware` added to `api.py`; `ALLOWED_HOSTS` env var configurable |

### Bug Fixes

| ID | Change |
|----|--------|
| B1 | `inference/vllm_server.py` — `is_ready()` returns `False` instead of raising when vLLM not started |
| B2 | `vl_trainer.py` — `_compute_vqa_accuracy` now raises `RuntimeError` if >50 % of samples fail (was silently returning 0.0) |
| B3 | `ft_platform/log_parser.py` — `parse_training_log` no longer crashes on malformed ANSI sequences |
| B4 | `inference/chat.py` — `prepare_inputs_for_generation` patch applied; prevents `NoneType.shape` crash in NemotronH VL inference |

### Structural Changes

| ID | Change |
|----|--------|
| R1 | Monolithic `app.py` (1500+ lines) split into `inference/`, `ft_platform/`, `ui/` packages |
| R2 | `DEV_USERNAME`/`DEV_PASSWORD` Gradio auth removed; Developer button routes directly to dev content (HTTP-level `ADMIN_PASSWORD` already guards the app) |
| R3 | CORS restricted: `allow_methods=["GET","POST","DELETE","OPTIONS"]`, `allow_headers=["Authorization","Content-Type","Accept"]`; Pydantic Field constraints on all request models |
| R4 | `platform/` renamed to `ft_platform/` to avoid shadowing the Python stdlib `platform` module |

### New Infrastructure

- **`Dockerfile`** — multi-stage NVIDIA CUDA 12.4.1 image; non-root `mkd` user
- **`docker-compose.yml`** — api + ui + redis services with GPU reservations
- **`Makefile`** — `test`, `lint`, `format`, `security`, `docker` targets
- **`.github/workflows/ci.yml`** — lint (black, ruff, mypy), security (bandit), test (pytest ≥70%) jobs
- **`pyproject.toml`** — PEP 621 package metadata replaces `setup.py`

### Documentation

- **`API_SPEC.md`** — `GET /api/eval/status` added; 422 validation errors documented; `hf_token` removed from publish spec
- **`FRONTEND_CONTRACT.md`** — `hf_token` removed from publish call example
- **`README.md`** — architecture diagram, security model, env vars table, migration guide
- **`PRODUCTION_CHECKLIST.md`** — pre-launch validation checklist

### Breaking Changes

1. `platform/` package renamed to `ft_platform/` — any existing import `from platform.log_parser import …` must become `from ft_platform.log_parser import …`
2. `POST /api/publish` no longer accepts `hf_token` in the request body — set the `HF_TOKEN` environment variable on the server
3. Developer Mode login screen removed — `ADMIN_PASSWORD` env var is the single auth gate
4. `ALLOWED_HOSTS` env var must be set in production (defaults to `localhost,127.0.0.1,*.mkd.kr`)
