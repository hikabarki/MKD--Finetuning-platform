# MKD Fine-Tuning Platform — Refactoring Roadmap

> Generated: 2026-06-19  
> Based on: full audit of app.py (4328 LOC), api.py (833), vl_trainer.py (694),
> pdf_utils.py (200), setup.py (38), scripts/ft (97), scripts/ft_add_dataset.py (710),
> scripts/ft_train.py (317), scripts/ft_clean_dataset.py (459), API_SPEC.md, FRONTEND_CONTRACT.md

---

## Executive Summary

The platform works in its current single-user demo context but has **six blockers** that prevent
production deployment, **two crash-level bugs**, and a **God-object app.py** that makes every
change risky. The refactoring should proceed in four phases over roughly six weeks — security
fixes first, then the structural split, then polish.

---

## Part 1 — Keep As-Is

These files and patterns are correct and do not need work before production.

| Component | Reason |
|---|---|
| `pdf_utils.py` overall structure | Clean fallback chain (pdf2image → pypdf), informative errors, per-page skip logic is solid |
| `scripts/ft_clean_dataset.py` pipeline ordering | schema → language → quality → length → PII → dedup → split is the right sequence |
| `scripts/ft_add_dataset.py` multi-source loader | HF / HTTP / local dispatch with graceful missing-dep degradation |
| `vl_trainer.py` loss-masking logic | Assistant-turn-only labels, image token handling, per-sample skip on error |
| `vl_trainer.py` VQA eval harness | Before/after accuracy comparison, sample-size warning, output via `VQA_METRICS=` |
| `api.py` SSE log-streaming pattern | Tail-thread with queue-based SSE output is idiomatic; no blocking |
| `api.py` single-active-job guard | `_active_job_lock` pattern is correct; just needs a TOCTOU fix |
| `scripts/ft` dispatcher | Simple `subprocess.call` dispatch is fine for a CLI tool; exit-code propagation works |
| `FRONTEND_CONTRACT.md` streaming snippets | The `streamLogs` / `streamChat` SSE patterns are production-quality JS |
| Chat `_build_chat_inputs` VL path | `apply_chat_template` → processor → pop `image_flags`/`num_patches` pipeline is correct |

---

## Part 2 — Minor Fixes

Issues that can be fixed in-place in under a day each, without structural changes.

### 2.1 Security (must fix before any external exposure)

**S1 — Path traversal on every user-supplied name (Critical)**  
Affects: `api.py` (all `name`/`run_id` params), `scripts/ft_add_dataset.py` (`--name`).  
Fix: add a single `_safe_name(s)` helper that rejects `/`, `..`, and null bytes, and call it at
every entry point before constructing a `Path`.
```python
def _safe_name(s: str) -> str:
    if not s or "/" in s or ".." in s or "\x00" in s:
        raise ValueError(f"Unsafe name: {s!r}")
    return s
```

**S2 — Hardcoded dev password fallback (Critical)**  
File: `app.py` line 1601.  
Fix: remove the default. If `DEV_PASSWORD` is unset, disable the dev tab entirely and log a
startup warning — do not fall back to a known string.
```python
DEV_PASSWORD = os.environ.get("DEV_PASSWORD")  # None means dev tab disabled
```

**S3 — Client-side-only developer auth (Critical)**  
File: `app.py`.  
Fix: add Gradio-level HTTP basic auth (`auth=` kwarg) for the whole app, keyed on
`ADMIN_PASSWORD`. The dev tab visibility toggle can stay as a UX convenience but cannot be
the access control layer. As a short-term fix, set `app.launch(auth=("admin", ADMIN_PASSWORD))`.

**S4 — HF token in request body / logs (High)**  
Files: `api.py` `PublishRequest`, `API_SPEC.md`.  
Fix: remove `hf_token` from `PublishRequest`. Read it from `HF_TOKEN` env var server-side.
Update `API_SPEC.md` and `FRONTEND_CONTRACT.md` to document that the token must be
configured on the server, not sent per-request.

**S5 — `trust_remote_code=True` without hash pinning (High)**  
File: `vl_trainer.py` lines 504, 542.  
Fix: add `revision=` (a pinned commit SHA) when loading the Nemotron model. Document the
expected SHA in a config constant so operators can audit it. Low effort, high value.

**S6 — World-readable debug logs containing pixel values (Medium)**  
Files: `app.py` lines 272–277, 446–448, 457–461.  
Fix: write debug logs to `logs/vl_debug.log` inside the project directory (not `/tmp`), and
strip or truncate the `[text]` line so raw model input is not persisted. Remove the
`[inputs]` and `[text]` lines entirely in non-DEBUG mode.

### 2.2 Crashes / Bugs (fix before any training run with validation data)

**B1 — `NameError: device` in `vl_trainer.py` (Crash)**  
File: `vl_trainer.py` line 646–647. `device` is referenced but never assigned in `main()`.
Only `device_map` is set.  
Fix: add `device = "cuda" if torch.cuda.is_available() else "cpu"` at the top of `main()`.

**B2 — `tempfile.mktemp()` race in `ft_train.py` (Medium risk)**  
File: `scripts/ft_train.py` line 224.  
Fix: replace with `tempfile.NamedTemporaryFile(suffix=".yaml", prefix="ft_train_",
delete=False)`.

**B3 — Bare `json.JSONDecodeError` in `ft_clean_dataset.py` (Crash on bad data)**  
File: `scripts/ft_clean_dataset.py` `load_jsonl` line 37.  
Fix: wrap the per-line parse in a `try/except` with a line-number counter and `sys.exit` on
more than N errors.

**B4 — Empty train set after filtering is not caught (Silent failure)**  
File: `scripts/ft_clean_dataset.py`.  
Fix: after `split_train_val`, assert `len(train) > 0` and exit with a clear diagnostic
message listing which filter removed the most rows.

### 2.3 Concurrency (fix before multi-user deployment)

**C1 — Locks declared but never acquired in `app.py` (Race condition)**  
Lines 38–40: `_vllm_lock`, `_user_model_path_lock`, `_user_training_lock` exist but are never
used. Meanwhile `_chat_model`, `_vllm_proc`, `_user_error_log_lines` are mutated from multiple
Gradio callback threads (pool size 10).  
Fix: wrap all reads/writes of these globals in `with _vllm_lock:` / `with _user_model_path_lock:`
as appropriate. This is a two-hour fix but prevents intermittent production crashes.

**C2 — TOCTOU in `api.py` training start (Low risk at single worker)**  
File: `api.py` lines 425–477. The lock is released between the check and the assignment.  
Fix: check and assign under the same lock acquisition.

**C3 — SSE stream holds file handle on client disconnect**  
File: `api.py` `_stream` generator.  
Fix: add `if await request.is_disconnected(): break` inside the yield loop.

### 2.4 Data quality

**D1 — Empty assistant labels in `pdf_utils.py` (Silent broken training)**  
File: `pdf_utils.py` line 133. `"content": ""` for every assistant turn.  
Fix: print a clear warning at the end of `build_vl_jsonl`:
`"⚠ Assistant labels are empty — edit the output JSONL before training."` and write a
`_NEEDS_ANNOTATION: true` field into the data card.

**D2 — Presidio English-only for Korean datasets**  
File: `scripts/ft_clean_dataset.py`.  
Fix: detect dataset language from the cleaning run's `--lang` argument and pass it to
`presidio_analyzer.AnalyzerEngine.analyze(language=...)`. Fall back to regex for `ko`.

**D3 — Fallback YAML parser silently drops nested keys**  
File: `scripts/ft_train.py` lines 48–55.  
Fix: the fallback exists only because PyYAML was once optional. `pyyaml` is now a standard
dep. Remove the hand-rolled fallback; add `pyyaml` to `install_requires` in `setup.py`.

### 2.5 Documentation gaps

**Doc1 — `API_SPEC.md` eval has no progress polling endpoint**  
Add `GET /api/eval/status` to both docs, or document that eval is synchronous and
`POST /api/eval/start` blocks until completion.

**Doc2 — `API_SPEC.md` error table: 202 listed under errors**  
Change the `202` row description from "Not an error — job started, begin polling" to a note
in the success section.

**Doc3 — `setup.py` personal email and no version pins**  
Replace `hikabarkif143@gmail.com` with a team/project address. Add minimum version pins
(`transformers>=4.40`, `peft>=0.10`, etc.) to prevent silent breakage.

---

## Part 3 — Major Rewrites

Work that requires structural changes and more than a day of effort.

### R1 — Split `app.py` into modules (Highest priority structural work)

**Problem:** 4,328 lines — CSS, business logic, ML inference, subprocess management, Gradio
layout, and UI callbacks all in one file. Every change risks a regression in an unrelated area.
No unit tests are possible.

**Target structure:**
```
ft-platform/
├── ui/
│   ├── __init__.py
│   ├── layout.py          # gr.Blocks() skeleton, top-level layout only
│   ├── screens/
│   │   ├── user_quickstart.py   # General User Mode, Quick Start
│   │   ├── user_custommix.py    # General User Mode, Custom Mix
│   │   └── developer/
│   │       ├── datasets.py      # Tabs 1-2
│   │       ├── training.py      # Tabs 3-5
│   │       └── eval_export.py   # Tabs 6-8
│   └── components.py     # Shared widgets, terminal_wrap, build_progress_html
├── inference/
│   ├── chat.py            # _load_chat_model, _build_chat_inputs, _generate_reply
│   ├── vllm_server.py     # _start_vllm_server, _stop_vllm_server
│   └── state.py           # Shared globals with proper lock acquisition
├── platform/
│   ├── log_parser.py      # read_log, parse_training_log, get_metrics_html
│   └── eval_dashboard.py  # auto-registration, dashboard data
└── app.py                 # Entry point: imports ui/layout.py, calls .launch()
```

**Key extractions:**
- `_generate_reply` + `_build_chat_inputs` + `_load_chat_model` → `inference/chat.py`
- `run_user_quickstart` and `run_user_custommix` share ~150 lines of VL subprocess code →
  extract `_run_vl_training_subprocess(cmd, run_id, output_fn)` to `platform/training.py`
- All `gr.Blocks()` content after line 1600 → `ui/` subpackage

**Effort:** 3–5 days. No new features, pure move.

### R2 — Replace developer login with real auth

**Problem:** The current implementation is security theater. Gradio's native
`auth=` kwarg handles HTTP Basic Auth at the server level. The developer tab needs to be
gated by a separate auth, not a Gradio callback that only hides DOM elements.

**Target:** Two-tier auth:
1. `app.launch(auth=[(ADMIN_USER, ADMIN_PASSWORD)])` for the dev-only Gradio instance (port
   7861, internal-only).
2. The general-user mode runs on port 7860 with no auth (or a simple shared password).

Alternatively: remove the dev UI from `app.py` entirely and have developers use the FastAPI
Swagger UI at `/api/docs`.

**Effort:** 1 day.

### R3 — Make `api.py` production-ready for multi-worker deployment

**Problem:** `_active_job`, `_inf_model`, `_inf_tok`, `_inf_is_vl` are process-local globals.
Running `uvicorn api:app --workers 2` causes each worker to have its own state; the frontend
talks to a random worker per request.

**Target:** Replace process-local state with either:
- A Redis-backed job registry (recommended for production), or
- Single-worker uvicorn with `--workers 1` and process-level concurrency via `asyncio` (acceptable for single-server deployments with explicit `--workers 1` documentation).

Also: add request-body size limits via `fastapi.middleware`:
```python
app.add_middleware(TrustedHostMiddleware, ...)
# or limit image_base64 at model level:
class ChatRequest(BaseModel):
    image_base64: Optional[str] = Field(None, max_length=10_000_000)  # ~7.5 MB
```

**Effort:** 2–3 days (Redis approach) or 1 day (single-worker with docs).

### R4 — `vl_trainer.py`: remove RCE surface, fix device scope

**Problem:** `trust_remote_code=True` combined with `_ensure_vl_cache` copying `.py` files
without verification creates a supply-chain attack surface. The `device` NameError crashes
any run that has a `val.jsonl`.

**Target:**
1. Pin model revision SHA in a config constant.
2. Replace `_ensure_vl_cache` with a hash-verified copy (check SHA256 of each `.py` against a
   known-good manifest).
3. Fix `device` scope bug (2-line fix).
4. Add a counter to the `except Exception: continue` in `_compute_vqa_accuracy` so that
   if >50% of samples fail, the function raises instead of silently returning `0.0`.

**Effort:** 1 day.

---

## Part 4 — Dependency Map and Refactoring Order

```
Phase 1 — Security & Crashes (Week 1, no structural change)
  S1  path traversal fix (api.py + ft_add_dataset.py)
  S2  remove hardcoded DEV_PASSWORD default
  S3  add Gradio HTTP auth (app.launch auth=)
  S4  remove hf_token from request body
  B1  fix NameError: device in vl_trainer.py
  B3  fix bare JSONDecodeError in ft_clean_dataset.py
  C1  activate declared locks in app.py
        ↓ unblocks any external exposure

Phase 2 — Data Quality & Minor Bugs (Week 2)
  D1  pdf_utils empty assistant labels warning
  D2  Presidio language fix (ft_clean_dataset.py)
  D3  remove fallback YAML parser, pin pyyaml
  B2  tempfile.mktemp → NamedTemporaryFile
  B4  zero-row guard after filtering
  C2  TOCTOU fix in api.py training start
  C3  SSE disconnect handling
  S6  move debug logs out of /tmp
        ↓ unblocks reliable training runs

Phase 3 — Structural Split of app.py (Weeks 3–4)
  R1  split app.py into ui/ + inference/ + platform/
    → depends on Phase 1 (globals need locks before being moved)
    → R2 (real auth) should be designed alongside this split
  R2  two-tier Gradio auth
        ↓ unblocks unit testing of business logic

Phase 4 — API & Trainer Hardening (Weeks 5–6)
  R3  api.py multi-worker / Redis job registry
    → depends on Phase 1 (path traversal fixed before expanding access)
  R4  vl_trainer.py RCE surface + device fix
    → B1 (device bug) must already be fixed from Phase 1
  S5  pin trust_remote_code revision SHA
  Doc1-3  update API_SPEC.md + FRONTEND_CONTRACT.md
        ↓ production-ready
```

### Dependency summary

| Work item | Blocked by | Blocks |
|---|---|---|
| Any external user access | S1, S2, S3 | — |
| Reliable VL training runs | B1, D3 | R4 |
| Multi-user concurrent chat | C1 | R1 |
| `app.py` split (R1) | C1 (locks must exist before move) | R2 |
| `api.py` multi-worker (R3) | S1 (path traversal), S4 (token) | frontend team production integration |
| Frontend team integration | Doc1-3, S4 | — |

---

## Risk Summary by Component

| File | Risk level | Effort to fix | First-priority action |
|---|---|---|---|
| `app.py` | 🔴 High (auth bypass, race, God object) | Large (R1, R2, C1) | C1 (activate locks) this sprint |
| `api.py` | 🔴 High (path traversal, token leak) | Medium (S1, S4, R3) | S1 this sprint |
| `vl_trainer.py` | 🟠 Medium (crash + RCE surface) | Small (B1 = 1 line) | B1 immediately |
| `scripts/ft_train.py` | 🟡 Low-medium (mktemp, YAML parser) | Small | D3 (remove fallback) |
| `scripts/ft_add_dataset.py` | 🟡 Low-medium (name path traversal) | Small | S1 applies here too |
| `scripts/ft_clean_dataset.py` | 🟡 Low-medium (Presidio, 0-row) | Small | B4, D2 |
| `pdf_utils.py` | 🟢 Low (empty labels) | Trivial | D1 (warning message) |
| `setup.py` | 🟢 Low (no pins, email) | Trivial | Doc3 |
| `API_SPEC.md` | 🟡 Low-medium (gaps) | Small | Doc1-2 |
| `FRONTEND_CONTRACT.md` | 🟢 Low (gaps only) | Trivial | Doc1 |
