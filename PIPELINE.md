# End-to-End Fine-Tuning Pipeline

**Platform:** MKD Self-Service Fine-Tuning — `ft` CLI  
**Maintainer:** Hika (MKD Co., Ltd.)  
**Target audience:** both engineers reading the code and non-engineers running commands

---

## Overview

The pipeline is a **linear sequence of 9 commands**. Each command does exactly one
job, saves its outputs, and prints the next command to run. You never need to
remember what comes next — the terminal tells you.

```
ft add-dataset  →  ft clean-dataset  →  ft add-model  →  ft configure
     ↓                    ↓                   ↓                ↓
  download &           dedupe,            download &       YAML recipe
  normalize           PII scrub,         verify           per GPU
  to ChatML           split              template

ft smoke-test  →  ft train  →  ft eval  →  ft export  →  ft publish
     ↓                ↓            ↓            ↓              ↓
  7 pre-flight    Axolotl +    lm-eval +    LoRA merge    HuggingFace
  checks +        watchdog     contam       + AWQ quant   + GitHub
  50-step run     abort        check        MODEL_CARD    push
```

Every successful command ends by printing a green box:

```
╭──────────────────────────────────────╮
│  NEXT STEP — copy and run this:      │
├──────────────────────────────────────┤
│  ft clean-dataset alpaca             │
╰──────────────────────────────────────╯
```

---

## Folder Layout

```
ft-platform/
├── README.md                    high-level intro
├── PIPELINE.md                  ← this file
├── STATE.md                     current phase + next action (auto-updated)
├── DECISIONS.md                 append-only architectural decision log
├── 01_scope.md … 07_handoff.md  per-phase engineering docs
├── install.sh                   one-time setup script
├── setup.py                     pip-installable package
│
├── scripts/                     all CLI scripts (source of truth)
│   ├── ft                       unified entry point  →  ~/.local/bin/ft
│   ├── ft_common.py             shared utilities (color printer, STATE updater)
│   ├── ft_add_dataset.py
│   ├── ft_clean_dataset.py
│   ├── ft_add_model.py
│   ├── ft_configure.py
│   ├── ft_smoke_test.py
│   ├── ft_train.py
│   ├── ft_eval.py
│   ├── ft_export.py
│   └── ft_publish.py
│
├── configs/                     training recipes (YAML) + eval suite config
│   ├── recipe_default.yaml      reference template (Axolotl-compatible)
│   └── eval_default.json        eval suite definition
│
├── guides/
│   └── 00_quickstart.md         non-engineer walkthrough
│
├── data_cards/                  one JSON provenance card per dataset version
├── datasets/                    downloaded + processed data  (gitignored raw)
├── models/                      downloaded base model weights (gitignored)
├── checkpoints/                 training output + crash dumps
├── eval_results/                immutable, timestamped eval artifacts
└── logs/                        training logs, NCCL traces, post-mortems
```

---

## Setup

### Requirements

| Item | Minimum |
|------|---------|
| OS | Ubuntu 22.04+ (or WSL2 on Windows) |
| GPU | NVIDIA, 24 GB VRAM minimum |
| Driver | ≥ 555 (for H100 / FlashAttention) |
| Python | 3.10+ |
| Disk | 200 GB free recommended |

### Installation

```bash
# 1. Clone or extract the platform
cd ~/fine-tune/ft-platform

# 2. Run the installer (detects GPU, installs torch + deps, registers ft on PATH)
./install.sh

# 3. Open a new terminal (or source ~/.bashrc), then verify:
ft --help
```

`install.sh` does the following:

1. Uses `~/fine-tune/venv` if it exists; otherwise creates `.venv/` inside the platform folder.
2. Detects your CUDA version via `nvidia-smi` and installs the matching PyTorch wheel (`cu124` for CUDA 12, `cu118` for CUDA 11).
3. Installs all platform dependencies: `huggingface_hub`, `datasets`, `transformers`, `accelerate`, `peft`, `trl`, `bitsandbytes`, `datasketch`, `pyyaml`, and more.
4. Creates a symlink `~/.local/bin/ft → scripts/ft`.
5. Appends `FT_PROJECT` and `PATH` to `~/.bashrc` / `~/.zshrc`.

After install, the environment variable `FT_PROJECT` points to this directory.
All scripts use it to locate `datasets/`, `models/`, `configs/`, etc.

---

## Step 1 — `ft add-dataset`

**Script:** [scripts/ft_add_dataset.py](scripts/ft_add_dataset.py)

```bash
ft add-dataset https://huggingface.co/datasets/tatsu-lab/alpaca
ft add-dataset tatsu-lab/alpaca          # short HF ID also works
ft add-dataset ./mydata.jsonl            # local file
ft add-dataset https://example.com/data.jsonl --name my-dataset
```

### What it does

1. **Classifies the source** — HuggingFace dataset ID, HTTP URL, or local path.
2. **Checks disk** — requires ≥ 5 GB free before downloading.
3. **Downloads** the data:
   - HuggingFace: uses `datasets.load_dataset()`, picks the `train` split (or first available split).
   - HTTP URL: streams to `datasets/<name>/raw/<filename>`.
   - Local file: reads directly.
4. **Auto-detects the input format** and converts to **ChatML JSONL**:

   | Input format | Detected by | Conversion |
   |---|---|---|
   | ChatML (`messages` column) | column name | passthrough |
   | Alpaca (`instruction`, `output`) | column names | system from instruction, user from input, assistant from output |
   | ShareGPT (`conversations` w/ `from`/`value`) | column names | `human`→user, `gpt`→assistant |
   | OpenAI (`prompt`, `completion`) | column names | wrap into 2-turn |
   | DPO (`prompt`, `chosen`, `rejected`) | column names | passthrough |
   | CSV / Parquet / JSONL | file extension | infer columns, route above |

5. **Writes outputs:**
   ```
   datasets/<name>/raw/          ← exact downloaded files
   datasets/<name>/normalized.jsonl  ← ChatML JSONL, one row per line
   data_cards/<name>.json        ← provenance card (source, format, row count, SHA-256)
   ```

6. **Warns if license is unknown** — the data card's `license` field is set to `"UNKNOWN — fill in before cleaning"`. The clean step will work but you should set it manually.

7. **Updates STATE.md** and prints:
   ```
   ft clean-dataset <name>
   ```

### Canonical ChatML format (SFT)

```json
{"messages": [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": "What is the capital of France?"},
  {"role": "assistant", "content": "Paris."}
]}
```

### Canonical DPO format

```json
{
  "prompt":   [{"role": "user", "content": "..."}],
  "chosen":   {"role": "assistant", "content": "..."},
  "rejected": {"role": "assistant", "content": "..."}
}
```

---

## Step 2 — `ft clean-dataset`

**Script:** [scripts/ft_clean_dataset.py](scripts/ft_clean_dataset.py)

```bash
ft clean-dataset alpaca
ft clean-dataset alpaca --max-chars 16384   # stricter length limit
ft clean-dataset alpaca --val-ratio 0.05    # larger validation set
ft clean-dataset alpaca --no-dedup          # skip dedup (tiny test sets only)
ft clean-dataset alpaca --no-pii            # skip PII scrub (NOT recommended)
```

### Pipeline (8 stages in order)

Each stage logs row counts in/out to `data_cards/<name>.json`.

| Stage | What happens |
|-------|-------------|
| 1. Schema validation | Drop rows missing required fields or malformed `messages` lists |
| 2. Length filter | Drop rows where total content > `--max-chars` (default 32,768 chars ≈ 8k tokens) |
| 3. PII scrub | Regex scan: **mask** emails → `[EMAIL]`, phones → `[PHONE]`, SSNs → `[SSN]`; **drop** rows with credit-card-like numbers |
| 4. MinHash dedup | LSH deduplication with `num_perm=128`, Jaccard threshold 0.85; falls back to exact SHA-1 dedup if `datasketch` is not installed |
| 5. Decontamination note | Full 13-gram overlap check is deferred to `ft eval`; recorded in data card |
| 6. Train/val split | Default 99/1 ratio, seeded at 42 |
| 7. Write outputs | `train.jsonl`, `val.jsonl`, `sample_100.jsonl` |
| 8. Update data card | Row counts at every stage, SHA-256 of train + val, PII report path |

### Outputs

```
datasets/<name>/train.jsonl          ← training set
datasets/<name>/val.jsonl            ← validation set
datasets/<name>/sample_100.jsonl     ← random 100-row sample for human review
data_cards/<name>.json               ← updated provenance card
eval_results/<name>_pii_report.json  ← {email, phone, ssn masked counts, dropped_cc}
```

### Human review step

After cleaning, open the sample and read the first few rows:

```bash
head -3 datasets/alpaca/sample_100.jsonl
```

If the conversations look sensible, proceed. If they look garbled or empty, run
`ft inspect-dataset <name>` or show the sample to your engineer before continuing.
This is the only manual step in the pipeline — 30 seconds here prevents hours of
wasted training.

---

## Step 3 — `ft add-model`

**Script:** [scripts/ft_add_model.py](scripts/ft_add_model.py)

```bash
ft add-model meta-llama/Llama-3.1-8B-Instruct
ft add-model https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
ft add-model meta-llama/Llama-3.1-8B-Instruct --accept-license
ft add-model meta-llama/Llama-3.1-8B-Instruct --revision v1.0
```

### What it does

1. **Parses the model ID** — full HuggingFace URL or `owner/name` short form.
2. **Checks disk** — requires ≥ 30 GB free (estimate; 8B model is ~16 GB).
3. **License probe** — queries HF Hub metadata. If the license contains any of
   `research`, `non-commercial`, `rail`, `cc-by-nc`, `llama2-community`,
   `llama3-community`, `gemma-terms-of-use`, or `other`, the download is
   **refused unless you pass `--accept-license`**. This protects against
   accidentally using restricted weights in a commercial product.
4. **Downloads weights** via `huggingface_hub.snapshot_download`:
   - Patterns: `*.safetensors`, `*.json`, `*.txt`, `tokenizer*`, `*.model`
   - Destination: `models/<owner>__<name>/`
5. **Chat-template round-trip test** — the #1 silent killer of fine-tuning runs.
   Renders a 3-turn sample conversation through `tokenizer.apply_chat_template()`,
   tokenizes it, decodes it back, and asserts every message content survives.
   Aborts if any content is lost. Warns if no `chat_template` is present on the
   tokenizer.
6. **Parameter count estimate** — derives approximate total parameters from
   `config.json` (hidden size, layers, vocab, FFN width, GQA heads). Accurate to
   within ~5% for standard decoder-only architectures.
7. **Memory budget table** — prints per-GPU peak VRAM for five training profiles:

   | Profile | Bytes/param | 8B example |
   |---------|-------------|-----------|
   | Full SFT bf16 + AdamW | ~16 | ~128 GB |
   | LoRA r=32, bf16 base | ~2.04 | ~16 GB |
   | QLoRA r=32 (NF4 + 8-bit AdamW) | ~0.53 | ~4.2 GB |
   | Inference bf16 | 2 | ~16 GB |
   | Inference 4-bit AWQ | 0.5 | ~4 GB |

8. **Writes `models/<name>/MODEL_INFO.json`:**
   ```json
   {
     "model_id": "meta-llama/Llama-3.1-8B-Instruct",
     "revision": "main",
     "local_path": "models/meta-llama__llama-3.1-8b-instruct",
     "license": "llama3-community",
     "params_billion": 8.030,
     "memory_estimates_gb": {...},
     "chat_template_present": true,
     "architecture": "LlamaForCausalLM"
   }
   ```
9. **Updates STATE.md** and prints:
   ```
   ft configure --model llama-3.1-8b-instruct --dataset alpaca
   ```

---

## Step 4 — `ft configure`

**Script:** [scripts/ft_configure.py](scripts/ft_configure.py)

```bash
ft configure --model llama-3.1-8b-instruct --dataset alpaca
ft configure --model llama-3.1-8b-instruct --dataset alpaca --quantize none  # full LoRA
ft configure --model llama-3.1-8b-instruct --dataset alpaca --lr 1e-4        # lower LR
ft configure --model llama-3.1-8b-instruct --dataset alpaca --epochs 5 --seq-len 4096
```

### What it does

1. **Loads `data_cards/<dataset>.json`** — verifies the dataset has been cleaned.
2. **Loads `models/<model>/MODEL_INFO.json`** — gets param count and local path.
   Falls back gracefully if the model is not downloaded locally (uses HF ID directly).
3. **Detects your GPU** via `nvidia-smi` — queries name, VRAM, and count.
4. **Selects a hardware profile:**

   | VRAM | micro_batch | grad_accum | optimizer | effective batch |
   |------|-------------|------------|-----------|----------------|
   | ≥ 70 GB (H100/A100) | 8 | 4 | paged_adamw_8bit | 32 |
   | ≥ 40 GB (A6000/etc) | 4 | 8 | paged_adamw_8bit | 32 |
   | < 40 GB (RTX 4090/3090) | 1 | 16 | adamw_torch | 16 |
   | CPU (testing only) | 1 | 16 | paged_adamw_8bit | 16 |

   If `--quantize none` is passed, uses full LoRA (bf16 base). Otherwise uses
   QLoRA with NF4 4-bit base weights. Models > 13B on GPUs < 40 GB are
   automatically forced to 4-bit QLoRA.

5. **Auto-detects chat template** from the model ID:
   - `llama-3` / `llama3` → `llama3`
   - `qwen2` / `qwen-2` → `chatml`
   - `mistral` → `mistral`
   - `gemma` → `gemma`
   - everything else → `chatml` (with a warning)

6. **Writes `configs/run_<YYYY-MM-DD_HHMM>.yaml`** — a complete Axolotl-compatible
   recipe with all fields filled in. Key fixed settings:

   | Setting | Value | Rationale |
   |---------|-------|-----------|
   | `adapter` | `qlora` | NF4 base weights; 4× less VRAM than LoRA |
   | `lora_r` | 32 | Standard rank; captures most task-relevant directions |
   | `lora_alpha` | 64 | alpha = 2×r is the standard default |
   | `lora_dropout` | 0.05 | Mild regularization on adapters |
   | Target modules | q/k/v/o/gate/up/down proj | All attention + MLP = best coverage |
   | `learning_rate` | 2e-4 | LoRA tolerates 5–10× higher LR than full-FT |
   | `lr_scheduler` | cosine | Standard for LLM fine-tuning |
   | `warmup_ratio` | 0.03 | 3% warmup |
   | `adam_beta2` | 0.95 | LLM default (not the PyTorch default of 0.999) |
   | `bf16` | true | Stable mixed precision; no fp32 copy needed for LoRA |
   | `flash_attention` | true | FlashAttention-2 (requires CUDA ≥ 11.6) |
   | `gradient_checkpointing` | true | Saves ~30–50% VRAM; ~20% slower |
   | `save_steps` | 200 | Checkpoint every 200 steps |
   | `save_total_limit` | 3 | Keep only 3 most recent checkpoints |
   | `seed` | 42 | Reproducible runs |

7. **Updates STATE.md** and prints:
   ```
   ft smoke-test configs/run_<ts>.yaml
   ```

---

## Step 5 — `ft smoke-test`

**Script:** [scripts/ft_smoke_test.py](scripts/ft_smoke_test.py)

```bash
ft smoke-test configs/run_2026-05-26_1430.yaml
```

**Always run this before full training. It is 30 minutes that saves 30 hours.**

### 7 checks (in order)

| Check | What it verifies | Failure action |
|-------|------------------|----------------|
| 1. Disk space | ≥ 10 GB free in `checkpoints/` | Warn (does not abort) |
| 2. GPU detection | `nvidia-smi` — name, VRAM, driver version | Warn if no GPU found |
| 3. Tokenizer round-trip | `apply_chat_template()` → tokenize → decode → all content survives | **Abort** if content lost |
| 4. Dataloader label-mask | Last turn in every row is `assistant` (the token that gets trained on) | **Abort** if wrong role |
| 5. 50-step run | Shape check + simulated timing: seq_len × micro_batch × 50 steps | **Abort** if fails |
| 6. Checkpoint r/w | Write a test JSON to `output_dir`, read it back, delete it | **Abort** if I/O fails |
| 7. Summary report | Prints GPU info, tok/s, estimated ETA for full run | Informational only |

### What the report looks like

```
  ┌─────────────────────────────────────────────┐
  │          SMOKE TEST REPORT                  │
  ├─────────────────────────────────────────────┤
  │  GPU:    NVIDIA H100 80GB PCIe              │
  │  VRAM:   80 GB                              │
  │  Tok/s:  41200                              │
  │  ETA:    ~5.2h for full run                 │
  └─────────────────────────────────────────────┘
```

If all checks pass, prints:
```
ft train configs/run_<ts>.yaml
```
with a reminder to run inside `tmux` so training survives terminal disconnection.

If any check fails, the script exits with code 1 and does **not** print the next
step. Fix the issue and re-run `ft smoke-test` — do not skip to training.

---

## Step 6 — `ft train`

**Script:** [scripts/ft_train.py](scripts/ft_train.py)

```bash
tmux new -s train          # recommended: prevents terminal-disconnect abortion
ft train configs/run_2026-05-26_1430.yaml
```

### What it does

1. **Pre-flight checks** — verifies training data exists at the path in the recipe,
   checks ≥ 10 GB free in `checkpoints/`.
2. **Detects GPU count** via `nvidia-smi` and builds the appropriate launch command:
   - Single GPU → `python -m axolotl.cli.train <recipe.yaml>`
   - Multi-GPU → `accelerate launch --num_processes=N --mixed_precision=bf16 -m axolotl.cli.train <recipe.yaml>`
   - No GPU → adds `--no_cuda` (testing only)
3. **Writes `env_snapshot.txt`** to the checkpoint dir — a full `pip freeze` output
   for exact reproducibility.
4. **Launches the training process** (Axolotl) with `NCCL_DEBUG=INFO` and
   `PYTHONUNBUFFERED=1` in the environment.
5. **Streams stdout** in real time, writing every line to
   `logs/train_<run-id>.log`.
6. **Real-time watchdog** — monitors every logged step and aborts on:

   | Signal | Threshold | Response |
   |--------|-----------|----------|
   | Loss is NaN or Inf | 1 occurrence | Abort immediately, save crash dump |
   | `grad_norm` | > 10 for 5 consecutive steps | Abort, save crash dump |
   | Loss spike | > 5× rolling median for 3 steps | Abort, save crash dump |

7. **Crash dump** — on abort, saves the last 100 log lines and a prefilled
   `post_mortem.md` template to `checkpoints/<run-id>/crash_dump/`.
8. On clean finish, prints:
   ```
   ft eval checkpoints/run_<ts>
   ```

### Resuming an interrupted run

```bash
ft resume checkpoints/run_2026-05-26_1430/step_1200
```

The resume command verifies tokenizer SHA, dataset SHA, and optimizer state
before reloading. If any of these have changed since the interrupted run, it
refuses to resume (silent data changes are a common cause of corrupted
multi-stage runs).

### Monitoring while training

In a second terminal:

```bash
ft watch run_2026-05-26_1430
```

Prints rolling loss, `grad_norm`, tok/s, ETA, and a sample generation every
200 steps.

---

## Step 7 — `ft eval`

**Script:** [scripts/ft_eval.py](scripts/ft_eval.py)

```bash
ft eval checkpoints/run_2026-05-26_1430
ft eval checkpoints/run_2026-05-26_1430 --base meta-llama/Llama-3.1-8B-Instruct
ft eval checkpoints/run_2026-05-26_1430 --skip-harness   # contamination check only
ft eval checkpoints/run_2026-05-26_1430 --suite configs/eval_default.json
```

### Eval suite (default: `configs/eval_default.json`)

| Layer | Eval | Purpose |
|-------|------|---------|
| Capability | MMLU (5-shot) | Broad knowledge |
| Capability | ARC-Challenge (25-shot) | Reasoning |
| Capability | HellaSwag (10-shot) | Common sense |
| Capability | TruthfulQA (0-shot) | Factual accuracy |
| Math | GSM8K (5-shot) | Arithmetic reasoning |
| Instruction-following | IFEval (0-shot) | Follows instructions |

All harness evals run via `lm-evaluation-harness`
(`python -m lm_eval --model hf --model_args pretrained=<path> ...`).

### Contamination check

After harness evals, the script builds a set of 13-grams from a sample of up to
10,000 training rows and records how many unique n-grams exist. Full 13-gram
overlap against each eval set is computed and written to:

```
eval_results/<run-id>/contamination.json
```

`ft publish` will block if any eval has contamination ratio > 0.5% (configurable
in `eval_default.json` via `"contamination_threshold": 0.005`).

### A/B comparison vs base model

```bash
ft eval-vs-base checkpoints/run_<ts> \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --prompts datasets/alpaca/eval_prompts_500.jsonl \
    --judge gpt-4o
```

Outputs `eval_results/<run-id>/vs_base.json`:

```json
{
  "win_rate": 0.612,
  "tie_rate": 0.124,
  "lose_rate": 0.264,
  "wilcoxon_p": 0.0003
}
```

**Interpretation:** win rate < 0.55 with overlapping CIs = no meaningful
improvement. Consider adjusting data, learning rate, or epoch count before publishing.

### Immutable artifact tree

```
eval_results/<run-id>/
├── lm_eval_harness.json      ← raw lm-eval output
├── contamination.json        ← 13-gram overlap report
├── env_snapshot.txt          ← pip freeze at eval time
└── summary.md                ← human-readable digest
```

`eval_results/` is intentionally **not** in `.gitignore` — these results are
the permanent record of what the model can do and should be committed.

---

## Step 8 — `ft export`

**Script:** [scripts/ft_export.py](scripts/ft_export.py)

```bash
ft export checkpoints/run_2026-05-26_1430 --merge --quantize awq
ft export checkpoints/run_2026-05-26_1430 --merge --quantize none   # bf16 only
ft export checkpoints/run_2026-05-26_1430 --merge --gguf            # also produce GGUF
```

### What it does

1. **Finds the base model** — reads `config.json`, `axolotl_config.yaml`, or
   `training_args.json` in the checkpoint; falls back to scanning `configs/*.yaml`
   for a `base_model:` line.
2. **Merges LoRA adapters** into the base model using `peft.AutoPeftModelForCausalLM`:
   - Loads adapter on CPU.
   - Calls `.merge_and_unload()`.
   - Saves with `safe_serialization=True` (safetensors format).
   - Output: `exports/<run-id>/merged/`
3. **Quantizes to AWQ** (default) using `autoawq`:
   - Config: zero-point, group size 128, W4 (GEMM kernel).
   - Output: `exports/<run-id>/awq/`
   - Suitable for vLLM / SGLang serving.
4. **Copies tokenizer** — all `tokenizer*.json`, `tokenizer*.model`,
   `special_tokens*.json` files to `exports/<run-id>/tokenizer/`.
5. **Inference test** — runs 5 sample prompts through the exported model
   (`transformers` generation, `max_new_tokens=128`) and saves responses to
   `exports/<run-id>/inference_test.json`. This is a final sanity check that
   the model actually generates text.
6. **Generates `MODEL_CARD.md`** — a template with placeholders:
   ```
   exports/<run-id>/MODEL_CARD.md
   ```
   You **must** fill in the `{filled by user}` sections before publishing.
   `ft publish` blocks if any placeholder remains.

### Quantization options

| Format | Flag | Use case | Quality loss vs bf16 |
|--------|------|----------|----------------------|
| bf16 (no quant) | `--quantize none` | Research / max quality | 0 |
| AWQ 4-bit | `--quantize awq` (default) | vLLM / SGLang serving | < 1% on MMLU |
| GGUF Q4_K_M | `--gguf` | llama.cpp / on-device | ~1% |

### Export tree

```
exports/<run-id>/
├── merged/            ← LoRA merged into base (bf16, safetensors)
├── awq/               ← 4-bit AWQ
├── gguf/              ← Q4_K_M GGUF  (only if --gguf)
├── tokenizer/         ← tokenizer files
├── MODEL_CARD.md      ← EDIT THIS before ft publish
├── inference_test.json
└── README.md          ← copy of MODEL_CARD for HF display
```

---

## Step 9 — `ft publish`

**Script:** [scripts/ft_publish.py](scripts/ft_publish.py)

```bash
ft publish exports/run_2026-05-26_1430 \
    --hf-repo mkd-hika/llama-3.1-8b-alpaca-ko

ft publish exports/run_2026-05-26_1430 \
    --hf-repo mkd-hika/llama-3.1-8b-alpaca-ko \
    --github-repo mkd-hika/llama-3.1-8b-alpaca-ko

# Override contamination block (only if you've reviewed the report):
ft publish ... --acknowledge-contamination
```

### Pre-publish guards (all must pass)

| Guard | Check | Override flag |
|-------|-------|---------------|
| MODEL_CARD.md placeholders | No `{filled by user}` text remaining | None — must be filled |
| Contamination | No eval set has ratio > 0.5% | `--acknowledge-contamination` |
| Base license | Not research-only / NC | `--accept-base-license` |

### What gets pushed

**HuggingFace Hub** (`huggingface-cli upload`):
- Pushes the quantized model dir (`awq/` by default, or `merged/` if AWQ was skipped)
- Uploads `MODEL_CARD.md` as `README.md` for the HF model page
- Commit message: "Initial release"

**GitHub** (optional, `--github-repo`):
- Copies `scripts/`, `configs/`, and all `.md` / `.yaml` root files to a temp dir
- `git init` → `git add .` → `git commit` → `git push -u origin main`
- Does **not** push model weights (those stay on HF)

### Final output

```
🎉 Published.
   Weights:    https://huggingface.co/mkd-hika/llama-3.1-8b-alpaca-ko
   Code:       https://github.com/mkd-hika/llama-3.1-8b-alpaca-ko
   Model card: https://huggingface.co/mkd-hika/llama-3.1-8b-alpaca-ko/blob/main/README.md
```

---

## Common Error Reference

| Error message | Likely cause | Fix |
|---------------|-------------|-----|
| `CUDA out of memory` | Model too large for GPU | Re-run `ft configure --quantize 4bit` |
| `chat template round-trip FAILED` | Tokenizer mismatch or missing template | Pick a different base model, or add `chat_template:` manually to recipe |
| `License is restricted` | Research-only weights | Add `--accept-license` if your use complies |
| `Loss is NaN` | Training instability (LR too high, bad data) | Re-run `ft configure --lr 1e-4` |
| `grad_norm > 10 for 5 steps` | Same as NaN — instability | Lower LR; check for all-caps / repeated text rows in your dataset |
| `Not enough disk` | < 10 GB free | Free space; checkpoints need 20–50 GB per run |
| `Axolotl not found` | Trainer not installed | `pip install axolotl` |
| `MODEL_CARD.md has unfilled placeholders` | Forgot to edit MODEL_CARD | Open `exports/<run>/MODEL_CARD.md` and fill in `{...}` sections |
| `Contamination check failed` | Training data overlaps eval set | Review `eval_results/<run>/contamination.json`; if intentional, use `--acknowledge-contamination` |

---

## Key Files at Each Stage

| After running | Key file created |
|---------------|-----------------|
| `ft add-dataset` | `datasets/<name>/normalized.jsonl`, `data_cards/<name>.json` |
| `ft clean-dataset` | `datasets/<name>/train.jsonl`, `datasets/<name>/val.jsonl`, `datasets/<name>/sample_100.jsonl` |
| `ft add-model` | `models/<name>/MODEL_INFO.json` |
| `ft configure` | `configs/run_<ts>.yaml` |
| `ft smoke-test` | `checkpoints/<run>/smoke_test_marker.json` (deleted on success) |
| `ft train` | `checkpoints/<run>/`, `logs/train_<run>.log`, `checkpoints/<run>/env_snapshot.txt` |
| `ft eval` | `eval_results/<run>/lm_eval_harness.json`, `eval_results/<run>/contamination.json`, `eval_results/<run>/summary.md` |
| `ft export` | `exports/<run>/merged/`, `exports/<run>/awq/`, `exports/<run>/MODEL_CARD.md`, `exports/<run>/inference_test.json` |
| `ft publish` | Live model at `huggingface.co/mkd-hika/<repo>` |

---

## Architecture Decisions (summary)

Full ADRs are in [DECISIONS.md](DECISIONS.md). The key ones:

**ADR-001 — CLI-first:** A guided CLI with printed next-steps ships faster, is
easier to debug, and can be wrapped by a web UI later without changing the core
logic. Web UI is deferred to v2.

**ADR-002 — Axolotl as primary trainer:** YAML-driven, supports SFT/DPO/ORPO,
works with FSDP + DeepSpeed for multi-GPU. Unsloth is used as a fast-path
for single-GPU LoRA jobs (2× speed, 60% less VRAM). Final choice is formalized
in ADR-003 once hardware tier is confirmed.

**Why ChatML as the canonical format:** It maps cleanly to every modern chat
template (`tokenizer.apply_chat_template()` handles per-model rendering at
training time). On-disk data is model-agnostic; switching base models does not
require re-processing the dataset.

**Why QLoRA by default:** A 4-bit NF4 base + bf16 LoRA adapters runs an 8B
model in ~4.2 GB of VRAM. This makes the platform usable on a single RTX 3090
without any configuration changes. Quality loss vs full LoRA is < 1% on most benchmarks.

---

## v2 Roadmap (not in scope for v1)

- Web UI (wraps the same CLI scripts via subprocess)
- Multi-tenant job queue (sequential is fine for ≤5 users)
- SFT + DPO in a single pipeline invocation
- ORPO training method
- S3 / GCS storage backend for datasets and checkpoints
- Automatic hyperparameter search (Optuna)
