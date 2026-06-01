# Self-Service Fine-Tuning Platform

A guided, command-line fine-tuning pipeline designed so **non-engineers** can
take a base model + a dataset URL and ship a fine-tuned model — with the system
telling them exactly what command to run next at every step.

Maintained by: **Hika** (MKD Co., Ltd. — https://mkd.kr)

---

## What this is

A thin, opinionated orchestration layer on top of proven OSS tools
(`huggingface_hub`, `datasets`, `axolotl` / `LLaMA-Factory` / `trl`, `unsloth`,
`vllm`). Each step is a single script. Each script ends by printing **the exact
next command the user should run**, so the workflow is self-guiding.

## The user journey (what an end-user sees)

```
$ ft add-dataset https://huggingface.co/datasets/tatsu-lab/alpaca
  → downloads, inspects, normalizes to ChatML JSONL
  → prints: "✓ Done. Next: ft clean-dataset alpaca"

$ ft clean-dataset alpaca
  → deduplicates, decontaminates, filters by length, saves data_card
  → prints: "✓ Done. Next: ft add-model meta-llama/Llama-3.1-8B-Instruct"

$ ft add-model meta-llama/Llama-3.1-8B-Instruct
  → downloads weights + tokenizer, verifies chat template
  → prints: "✓ Done. Next: ft configure --model llama-3.1-8b --dataset alpaca"

$ ft configure --model llama-3.1-8b --dataset alpaca
  → generates configs/run_<timestamp>.yaml with sane defaults for detected GPU
  → prints: "✓ Done. Next: ft smoke-test configs/run_<ts>.yaml"

$ ft smoke-test configs/run_<ts>.yaml
  → 50-step dry run, validates everything fits in memory, no NaN
  → prints: "✓ Done. Next: ft train configs/run_<ts>.yaml"

$ ft train configs/run_<ts>.yaml
  → full training, with periodic eval and W&B logging
  → prints: "✓ Done. Next: ft eval checkpoints/<run-id>"

$ ft eval checkpoints/<run-id>
  → runs lm-eval-harness + a contamination check
  → prints: "✓ Done. Next: ft export checkpoints/<run-id> --merge --quantize awq"

$ ft export checkpoints/<run-id> --merge --quantize awq
  → merges LoRA, quantizes for serving, writes MODEL_CARD.md
  → prints: "✓ Done. Push to HF? Run: ft publish checkpoints/<run-id>"

$ ft publish checkpoints/<run-id>
  → pushes to huggingface.co/mkd-hika/<repo-name>
  → prints: "🎉 Live at https://huggingface.co/mkd-hika/<repo-name>"
```

## The 7-Phase Protocol (engineering docs)

| Phase | Doc | Purpose |
|-------|-----|---------|
| 1 | [01_scope.md](01_scope.md) | Intake & objectives |
| 2 | [02_architecture.md](02_architecture.md) | Method, parallelism, precision |
| 3 | [03_data.md](03_data.md) | Data pipeline & schema |
| 4 | [04_recipe.yaml](04_recipe.yaml) | Concrete training recipe |
| 5 | [05_runbook.md](05_runbook.md) | Run plan & abort criteria |
| 6 | [06_eval_plan.md](06_eval_plan.md) | Evaluation & contamination |
| 7 | [07_handoff.md](07_handoff.md) | Deployment & model card |

## Folder layout

```
ft-platform/
├── README.md                  ← you are here
├── STATE.md                   ← current phase, blockers, next action
├── DECISIONS.md               ← append-only ADR log
├── 01_scope.md … 07_handoff.md
├── configs/                   ← runnable YAML recipes
├── scripts/                   ← the ft-* CLI scripts (the user-facing UX)
├── data_cards/                ← one card per dataset version
├── datasets/                  ← downloaded & processed datasets (gitignored)
├── models/                    ← downloaded base models (gitignored)
├── checkpoints/               ← training output manifests
├── eval_results/              ← timestamped, never overwritten
├── logs/                      ← training logs, NCCL traces, post-mortems
└── guides/                    ← non-engineer-facing markdown manuals
```

## Where to start

If you are **the operator setting this up**: read `STATE.md`, then `01_scope.md`.
If you are **an end-user**: read `guides/00_quickstart.md`.
