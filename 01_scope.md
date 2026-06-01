# 01 — Scope & Intake

## Project
Self-service fine-tuning platform: a non-engineer pastes a dataset URL and a
HuggingFace model link; the system handles the rest, printing the next command
at every step.

## Goals
- **G1.** Anyone literate in copy-paste can fine-tune a model end-to-end.
- **G2.** Every script is idempotent and prints "what to run next."
- **G3.** Sensible defaults that work on the operator's chosen hardware tier.
- **G4.** All artifacts (configs, data cards, eval results) auto-saved into the
  project folder for reproducibility.
- **G5.** Optional one-command push to `huggingface.co/mkd-hika/<repo>` and
  `github.com/mkd-hika/<repo>`.

## Non-goals (v1)
- Web UI (deferred — see ADR-001)
- Multi-tenant job queue
- Custom CUDA kernels / from-scratch trainers
- Pretraining (this is a fine-tuning platform)

## Personas
- **End user (primary):** product manager, domain expert, junior researcher.
  Comfortable with terminal copy-paste. Not expected to write Python or YAML.
- **Operator (secondary):** the engineer (you / me) who installs, configures,
  monitors GPU health, debugs failures.

## Supported workflows (v1 target)
1. **SFT-only** (default) — instruction tuning on chat / instruction data
2. **SFT + DPO** — preference tuning after SFT
3. *(stretch)* **SFT + ORPO** — combined preference + SFT in one stage

## Supported data sources
- HuggingFace dataset URL (`https://huggingface.co/datasets/<owner>/<name>`)
- Direct file URL (`.jsonl`, `.json`, `.csv`, `.parquet`)
- Local path
- *(stretch)* Google Sheets export URL, S3 URI

## Open Questions — need user input before Phase 2

> **Hika needs answers to all 5 before writing `02_architecture.md`.**

- **[Q-001] Target hardware tier?**
  - (a) Single consumer GPU, 24 GB (RTX 4090 / 3090)
  - (b) Single H100 80 GB / A100 80 GB
  - (c) Single node, 4–8× H100
  - (d) Multi-node cluster
- **[Q-002] Model size envelope?**
  - (a) ≤7B (Qwen2.5-7B, Llama-3.1-8B class)
  - (b) 7–13B
  - (c) 30–70B (Llama-3.1-70B class)
  - (d) >70B (Mixtral-8x22B, DeepSeek-V3)
- **[Q-003] How many concurrent users / jobs do you expect?**
  - (a) 1 — solo
  - (b) 2–5 — small team, sequential is fine
  - (c) Many — needs job queue (deferred to v2 if so)
- **[Q-004] Which training methods does v1 need?**
  - (a) SFT only
  - (b) SFT + DPO
  - (c) SFT + DPO + ORPO
- **[Q-005] Where do datasets and checkpoints live?**
  - (a) Local disk only
  - (b) HuggingFace Hub (private repos under `mkd-hika`)
  - (c) S3 / GCS bucket (provide endpoint when answering)

## Answers
*(filled in by Hika after user responds; supersedes the questions above.)*

| Q | Answer | Notes |
|---|--------|-------|
| Q-001 | _pending_ | |
| Q-002 | _pending_ | |
| Q-003 | _pending_ | |
| Q-004 | _pending_ | |
| Q-005 | _pending_ | |

## Compliance & licensing reminders
- The pipeline must refuse to download base weights with research-only or
  non-commercial licenses unless the operator explicitly acknowledges via
  `--accept-license` flag.
- Dataset license is recorded in `data_cards/<dataset>.json`; if the field is
  missing, the dataset is quarantined until the operator fills it in.
- PII scan (presidio) runs in `ft clean-dataset`; flagged rows are dropped by
  default with a report at `eval_results/<dataset>_pii_report.json`.
