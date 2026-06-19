Read /mnt/agents/upload/REFACTORING_ROADMAP.md to understand the full plan. We are executing Phase 1: Security & Crash Fixes only in /home/mkd-s002/ft-platform-refactor/. Do NOT refactor structure — only patch in-place.

Execute these exact fixes from the roadmap:

1. S1 — Path traversal fix (Critical)
   - Files: /home/mkd-s002/ft-platform-refactor/api.py (all name/run_id params), /home/mkd-s002/ft-platform-refactor/scripts/ft_add_dataset.py (--name)
   - Add this helper and call it at every entry point before constructing Path:
   def _safe_name(s: str) -> str:
       if not s or "/" in s or ".." in s or "\x00" in s:
           raise ValueError(f"Unsafe name: {s!r}")
       return s
   - Apply to: dataset endpoints, model endpoints, training endpoints, eval endpoints, export endpoints in api.py. Also ft_add_dataset.py.

2. S2 — Remove hardcoded DEV_PASSWORD fallback (Critical)
   - File: /home/mkd-s002/ft-platform-refactor/app.py line ~1601
   - Change: DEV_PASSWORD = os.environ.get("DEV_PASSWORD", "mkd2026!") to DEV_PASSWORD = os.environ.get("DEV_PASSWORD")
   - If DEV_PASSWORD is None, disable the dev tab entirely and log a startup warning.

3. S3 — Add Gradio HTTP auth (Critical)
   - File: /home/mkd-s002/ft-platform-refactor/app.py
   - Add app.launch(auth=("admin", ADMIN_PASSWORD)) where ADMIN_PASSWORD is from env var.
   - The dev tab visibility toggle stays as UX convenience but cannot be the access control layer.

4. S4 — Remove hf_token from request body (High)
   - Files: /home/mkd-s002/ft-platform-refactor/api.py PublishRequest, /home/mkd-s002/ft-platform-refactor/API_SPEC.md
   - Remove hf_token field from PublishRequest Pydantic model.
   - Read HF token from HF_TOKEN env var server-side.
   - Update API_SPEC.md and FRONTEND_CONTRACT.md.

5. B1 — Fix NameError: device in vl_trainer.py (Crash)
   - File: /home/mkd-s002/ft-platform-refactor/vl_trainer.py lines ~646-647
   - Add device = "cuda" if torch.cuda.is_available() else "cpu" at top of main().

6. B3 — Fix bare JSONDecodeError in ft_clean_dataset.py (Crash)
   - File: /home/mkd-s002/ft-platform-refactor/scripts/ft_clean_dataset.py line ~37 (load_jsonl)
   - Wrap per-line parse in try/except with line-number counter. Exit if >N errors.

7. C1 — Activate declared locks in app.py (Race condition)
   - Lines 38-40: _vllm_lock, _user_model_path_lock, _user_training_lock exist but are never used.
   - Wrap all reads/writes of these globals in with _vllm_lock: / with _user_model_path_lock: etc.
   - _chat_model, _chat_tok, _chat_name, _chat_is_vision -> use _vllm_lock or create _chat_lock
   - _vllm_proc, _vllm_client, _vllm_model_id -> _vllm_lock
   - _user_last_model_path -> _user_model_path_lock
   - _user_training_proc -> _user_training_lock

For each fix:
- Show me the exact diff (before/after)
- Validate with python -m py_compile /home/mkd-s002/ft-platform-refactor/<file>
- Do NOT refactor anything else — minimal changes only

After all fixes, run: cd /home/mkd-s002/ft-platform-refactor && python -c "import app; print('Import OK')"

Report back which items are done and which need manual review.

---
Read /mnt/agents/upload/REFACTORING_ROADMAP.md. We are executing Phase 2: Data Quality & Minor Bugs only in /home/mkd-s002/ft-platform-refactor/.

Execute these fixes in-place (no structural changes):

1. D1 — pdf_utils empty assistant labels warning
   - File: /home/mkd-s002/ft-platform-refactor/pdf_utils.py line ~133
   - After build_vl_jsonl completes, print warning if all assistant labels are empty.
   - Write "_NEEDS_ANNOTATION": true to data card.

2. D2 — Presidio English-only for Korean datasets
   - File: /home/mkd-s002/ft-platform-refactor/scripts/ft_clean_dataset.py
   - Detect language from --lang argument, pass to presidio_analyzer.AnalyzerEngine.analyze(language=...).
   - Fall back to regex for Korean (ko).

3. D3 — Remove fallback YAML parser
   - File: /home/mkd-s002/ft-platform-refactor/scripts/ft_train.py lines ~48-55
   - Remove hand-rolled YAML fallback. Add pyyaml to install_requires in setup.py.

4. B2 — tempfile.mktemp() race
   - File: /home/mkd-s002/ft-platform-refactor/scripts/ft_train.py line ~224
   - Replace with tempfile.NamedTemporaryFile(suffix=".yaml", prefix="ft_train_", delete=False)

5. B4 — Zero-row guard after filtering
   - File: /home/mkd-s002/ft-platform-refactor/scripts/ft_clean_dataset.py
   - After split_train_val, assert len(train) > 0. Exit with diagnostic showing which filter removed most rows.

6. C2 — TOCTOU fix in api.py training start
   - File: /home/mkd-s002/ft-platform-refactor/api.py lines ~425-477
   - Check and assign _active_job under the same lock acquisition.

7. C3 — SSE disconnect handling
   - File: /home/mkd-s002/ft-platform-refactor/api.py _stream generator
   - Add if await request.is_disconnected(): break inside yield loop.

8. S6 — Move debug logs out of /tmp
   - Files: /home/mkd-s002/ft-platform-refactor/app.py lines ~272-277, ~446-448, ~457-461
   - Write to logs/vl_debug.log inside project directory. Strip raw model input in non-DEBUG mode.

For each fix:
- Show exact diff
- Validate with python -m py_compile /home/mkd-s002/ft-platform-refactor/<file>
- Update /home/mkd-s002/ft-platform-refactor/API_SPEC.md if behavior changes

After all fixes, run: cd /home/mkd-s002/ft-platform-refactor && python -c "import app; import api; print('All imports OK')"

Report back which items are done.

---
Read /mnt/agents/upload/REFACTORING_ROADMAP.md. We are executing Phase 3: Split app.py into Modules (R1) in /home/mkd-s002/ft-platform-refactor/.

Split the 4,328-line /home/mkd-s002/ft-platform-refactor/app.py into this exact structure:

/home/mkd-s002/ft-platform-refactor/
├── ui/
│   ├── __init__.py
│   ├── layout.py
│   ├── screens/
│   │   ├── __init__.py
│   │   ├── user_quickstart.py
│   │   ├── user_custommix.py
│   │   └── developer/
│   │       ├── __init__.py
│   │       ├── datasets.py
│   │       ├── training.py
│   │       └── eval_export.py
│   └── components.py
├── inference/
│   ├── __init__.py
│   ├── chat.py
│   ├── vllm_server.py
│   └── state.py
├── platform/
│   ├── __init__.py
│   ├── log_parser.py
│   ├── eval_dashboard.py
│   └── training.py
└── app.py

Extraction rules:
- inference/chat.py: _generate_reply, _build_chat_inputs, _load_chat_model
- inference/vllm_server.py: _start_vllm_server, _stop_vllm_server, _is_vllm_model
- inference/state.py: All global mutable state with locks from Phase 1 C1. Use RLock.
- ui/components.py: terminal_wrap, ansi_to_html, build_progress_html, get_gpu_html, get_pipeline_html, get_stats_html, get_guide_html, get_state_html, next_step_card, read_sample, read_data_card, read_config_file, read_log, read_eval_summary, read_model_card, save_model_card
- platform/log_parser.py: get_loss_df, parse_training_log, get_metrics_html
- platform/eval_dashboard.py: build_loss_chart, build_ppl_chart, build_summary_chart, build_metrics_df, load_eval_run, get_comparison_chart
- platform/training.py: Extract shared VL subprocess code from run_user_quickstart and run_user_custommix into _run_vl_training_subprocess(cmd, run_id, output_fn)
- ui/screens/user_quickstart.py: user_screen1, user_screen2a, user_screen3, user_screen4 + run_user_quickstart
- ui/screens/user_custommix.py: user_screen2b + run_user_custommix + get_model_choices, get_cm_dataset_choices, _auto_model_name
- ui/screens/developer/datasets.py: Tab 1 (Dataset) + Tab 2 (Clean)
- ui/screens/developer/training.py: Tab 3 (Model) + Tab 4 (Configure) + Tab 5 (Train)
- ui/screens/developer/eval_export.py: Tab 6 (Eval) + Tab 7 (Export & Publish) + Tab 8 (Chat)
- ui/layout.py: The gr.Blocks() skeleton, landing page, navigation wiring

CRITICAL: Maintain exact same Gradio component IDs (elem_id) and event handler wiring. UI must look identical.

After extraction:
- app.py must be under 200 lines
- Each module under 300 lines
- All functions under 50 lines (split if needed)
- Verify imports: cd /home/mkd-s002/ft-platform-refactor && python -c "from ui.layout import create_app; print('OK')"
- Launch test: cd /home/mkd-s002/ft-platform-refactor && timeout 10 python app.py || true

Work module by module. Commit after each module extraction. Report progress after each file.

---
Read /mnt/agents/upload/REFACTORING_ROADMAP.md. We are executing Phase 4: API & Trainer Hardening in /home/mkd-s002/ft-platform-refactor/.

Execute these items:

1. R2 — Replace developer login with real auth
   - Remove DEV_USERNAME/DEV_PASSWORD Gradio callback auth from /home/mkd-s002/ft-platform-refactor/app.py
   - Option B preferred: Remove dev UI from app.py entirely, developers use FastAPI Swagger at /api/docs

2. R3 — Make api.py production-ready for multi-worker
   - Option A (recommended): Redis-backed job registry. Replace _active_job, _inf_model, _inf_tok, _inf_is_vl process-local globals. Use redis-py for job state. Add request body limits: ChatRequest.image_base64 max_length=10_000_000
   - Add TrustedHostMiddleware, restrictive CORSMiddleware
   - Add Pydantic Field constraints on all request models

3. R4 — vl_trainer.py hardening
   - Pin model revision SHA in config constant
   - Replace _ensure_vl_cache with hash-verified copy (SHA256 manifest)
   - Add counter to _compute_vqa_accuracy: if >50% samples fail, raise instead of returning 0.0

4. S5 — Pin trust_remote_code revision SHA
   - Add REVISION_SHA constant to config
   - Pass revision=REVISION_SHA to from_pretrained calls

5. Doc1-3 — Update documentation
   - /home/mkd-s002/ft-platform-refactor/API_SPEC.md: Add GET /api/eval/status or document synchronous eval
   - Fix 202 error table entry
   - /home/mkd-s002/ft-platform-refactor/FRONTEND_CONTRACT.md: Update for S4 (no hf_token in request)
   - /home/mkd-s002/ft-platform-refactor/setup.py: Replace personal email, add version pins

6. Add production infrastructure:
   - /home/mkd-s002/ft-platform-refactor/Dockerfile (multi-stage)
   - /home/mkd-s002/ft-platform-refactor/docker-compose.yml (api + ui + redis)
   - /home/mkd-s002/ft-platform-refactor/Makefile: test, lint, format, security (bandit), docker
   - /home/mkd-s002/ft-platform-refactor/.github/workflows/ci.yml (pytest, black, ruff, mypy, bandit)

7. Final validation:
   - cd /home/mkd-s002/ft-platform-refactor && pytest tests/ -v --cov=ui --cov=inference --cov=platform --cov=api
   - Coverage >70%
   - cd /home/mkd-s002/ft-platform-refactor && bandit -r . -f json -o bandit-report.json
   - cd /home/mkd-s002/ft-platform-refactor && mypy ui/ inference/ platform/ api.py
   - Zero errors from all tools

Deliverables:
- All modified files with diffs
- Test results
- Lint/security scan results
- Updated /home/mkd-s002/ft-platform-refactor/README.md with architecture, deployment, env vars
- /home/mkd-s002/ft-platform-refactor/PRODUCTION_CHECKLIST.md with all items checked

---
Read /mnt/agents/upload/REFACTORING_ROADMAP.md. We are executing Phase 5: Testing & Validation in /home/mkd-s002/ft-platform-refactor/.

Create comprehensive tests for all modules created in Phases 1-4:

1. Unit tests for inference/ package
   - tests/inference/test_chat.py: Mock _load_chat_model, test _build_chat_inputs text path, test _build_chat_inputs VL path, test _generate_reply with mock model
   - tests/inference/test_vllm_server.py: Mock subprocess.Popen, test start_vllm_server success/failure, test stop_vllm_server cleanup
   - tests/inference/test_state.py: Test SessionState thread safety with concurrent threads, test lock acquisition under load

2. Unit tests for platform/ package
   - tests/platform/test_log_parser.py: Create sample log files, test parse_training_log, test get_metrics_html, test get_loss_df
   - tests/platform/test_eval_dashboard.py: Test chart generation with mock data, test build_metrics_df
   - tests/platform/test_training.py: Mock subprocess, test _run_vl_training_subprocess streaming

3. Unit tests for ui/ package
   - tests/ui/test_components.py: Test terminal_wrap, ansi_to_html, build_progress_html output
   - tests/ui/test_screens.py: Test screen creation functions return valid Gradio components

4. Integration tests
   - tests/integration/test_app_launch.py: Test app.py imports and basic launch (mock Gradio)
   - tests/integration/test_api_endpoints.py: Test FastAPI endpoints with TestClient
   - tests/integration/test_security.py: Test _safe_name rejects bad inputs, test auth middleware

5. Security tests
   - tests/security/test_path_traversal.py: Test _safe_name with "/../etc/passwd", "\x00", empty strings
   - tests/security/test_shell_injection.py: Test that all subprocess calls use shlex.quote
   - tests/security/test_auth.py: Test that dev tab is disabled without password, test auth bypass attempts

6. Performance tests
   - tests/performance/test_concurrent_chat.py: Simulate 10 concurrent chat requests, verify no race conditions
   - tests/performance/test_gpu_monitor.py: Test get_gpu_html caching (2s TTL)

Requirements:
- Use pytest with fixtures
- Mock external dependencies (subprocess, torch, transformers, nvidia-smi)
- Use tmp_path for file operations
- Use monkeypatch for env vars
- All tests must pass in under 60 seconds total
- Coverage report: cd /home/mkd-s002/ft-platform-refactor && pytest tests/ --cov=. --cov-report=term-missing

After writing tests, run them and fix any failures until all pass.

---
Read /mnt/agents/upload/REFACTORING_ROADMAP.md. We are executing Phase 6: Final Polish & Documentation in /home/mkd-s002/ft-platform-refactor/.

Finish the production-ready package:

1. Update /home/mkd-s002/ft-platform-refactor/setup.py
   - Convert to pyproject.toml (PEP 621)
   - Add all packages: ui, inference, platform, api
   - Pin dependency versions: transformers>=4.40, peft>=0.10, gradio>=4.0, etc.
   - Add extras_require for dev: pytest, black, ruff, mypy, bandit

2. Create /home/mkd-s002/ft-platform-refactor/README.md
   - Architecture diagram (ASCII art or mermaid)
   - Security model (auth flow, input validation)
   - Deployment instructions (docker, bare metal, k8s)
   - Environment variables reference table
   - Migration guide from old monolithic app.py

3. Create /home/mkd-s002/ft-platform-refactor/PRODUCTION_CHECKLIST.md
   Verify and check all items:
   - [ ] Zero global mutable state
   - [ ] All locks used properly
   - [ ] No hardcoded secrets
   - [ ] All inputs sanitized
   - [ ] All subprocess calls have timeouts
   - [ ] Zero bare except Exception: clauses
   - [ ] All functions under 50 lines
   - [ ] All modules under 300 lines
   - [ ] Type hints on all public APIs
   - [ ] pytest suite passes (>70% coverage)
   - [ ] Structured logging throughout
   - [ ] Health check endpoint
   - [ ] Graceful shutdown
   - [ ] CSS externalized
   - [ ] Dockerfile exists
   - [ ] CI/CD pipeline configured
   - [ ] Documentation complete

4. Create /home/mkd-s002/ft-platform-refactor/CHANGELOG.md
   - Document all changes from original app.py
   - Security fixes (S1-S6)
   - Bug fixes (B1-B4)
   - Structural changes (R1-R4)
   - Breaking changes for users

5. Final validation commands
   Run these and report results:
   cd /home/mkd-s002/ft-platform-refactor
   python -m py_compile app.py api.py
   pytest tests/ -v --tb=short
   black --check ui/ inference/ platform/ api.py
   ruff check ui/ inference/ platform/ api.py
   mypy ui/ inference/ platform/ api.py
   bandit -r . -ll

6. Create git tag
   cd /home/mkd-s002/ft-platform-refactor
   git add -A
   git commit -m "v1.0.0: Production-ready refactor"
   git tag -a v1.0.0 -m "Production-ready release"

Deliverables:
- All new/modified files
- Command output from all validation steps
- Checked PRODUCTION_CHECKLIST.md
- Tagged commit hash