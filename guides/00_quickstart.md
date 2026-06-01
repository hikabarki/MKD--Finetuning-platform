# Quickstart — Fine-Tune a Model in 7 Commands

**This guide assumes nothing.** If you can open a terminal and copy-paste, you can finish.

---

## Before you start

You need:
- A computer with an **NVIDIA GPU** (the more memory the better — 24 GB minimum)
- **Ubuntu 22.04+** (or another Linux), or Linux inside WSL2 on Windows
- About **2 hours of GPU time** for a small experiment

Your operator (engineer) should have already run `./install.sh` for you.
If `ft --help` works in your terminal, you're ready.

---

## The whole journey, end to end

After each command, the screen will show a **green box** telling you the next
command to type. You don't need to remember any of this — just follow the boxes.

### Step 1 — Add a dataset

Pick any dataset on HuggingFace. Example: the famous "alpaca" dataset.

```bash
ft add-dataset https://huggingface.co/datasets/tatsu-lab/alpaca
```

What you'll see:
- A progress bar while it downloads
- A line that says how many rows were found
- A green box: **`NEXT STEP — copy and run this:`**

### Step 2 — Clean the dataset

Copy the command from the green box. It will look like:

```bash
ft clean-dataset alpaca
```

What this does (don't worry about the details):
- Removes duplicate rows
- Masks emails, phone numbers, credit cards
- Splits into training and validation sets
- Saves a `sample_100.jsonl` for you to eyeball

**Important:** open `datasets/alpaca/sample_100.jsonl` in a text editor and read
the first few lines. Do they look like sensible conversations? If yes, continue.
If they look weird, stop and ask your engineer.

### Step 3 — Add a base model

Copy any model link from HuggingFace and paste it. Example:

```bash
ft add-model https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

What you'll see:
- A license check (some models are research-only — you'll be asked to acknowledge)
- A big download (15–30 minutes for an 8B model on a fast connection)
- A "chat template test" — this catches the #1 bug in fine-tuning
- A memory estimate, so you know what fits on your GPU

### Step 4 — Generate a recipe

```bash
ft configure --model llama-3.1-8b-instruct --dataset alpaca
```

This makes a file in `configs/`. You don't need to open it. The defaults are
chosen by Hika based on your GPU.

### Step 5 — Smoke test (always!)

```bash
ft smoke-test configs/run_<timestamp>.yaml
```

This runs **only 50 training steps** to check that:
- Everything fits in your GPU's memory
- The loss is going down (not NaN)
- Checkpoints can be saved and reloaded

If this fails, **do not skip to training.** Show the error to your engineer.

### Step 6 — Train

This is the long one (hours).

```bash
tmux new -s train        # so it survives if your terminal closes
ft train configs/run_<timestamp>.yaml
```

In a separate terminal, watch progress:
```bash
ft watch run_<timestamp>
```

You'll see loss going down, GPU usage, an ETA, and a sample generation every
few hundred steps. If anything goes wrong, the script aborts and saves
diagnostics into `logs/`.

### Step 7 — Evaluate and publish

```bash
ft eval checkpoints/run_<timestamp>
ft export checkpoints/run_<timestamp>
ft publish exports/run_<timestamp>
```

The publish command:
- Uploads weights to `https://huggingface.co/mkd-hika/<your-repo-name>`
- Pushes your training code to `https://github.com/mkd-hika/<your-repo-name>`
- Prints both URLs at the end. **You're done.**

---

## When things go wrong

The most common errors and what they mean:

| Error message contains | Likely cause | What to do |
|------------------------|--------------|-----------|
| `CUDA out of memory` | Model too big for your GPU | Re-run `ft configure --quantize 4bit` |
| `chat template round-trip FAILED` | Tokenizer mismatch | Pick a different base model, or ask engineer |
| `License is restricted` | Research-only weights | Re-run `ft add-model ... --accept-license` if your use case complies |
| `Loss is NaN` | Training instability | Re-run `ft configure --lr 1e-4` (half the learning rate) |
| `Disk full` | Not enough space | Free up space; checkpoints need 30+ GB |

For anything else: take a screenshot of the last 30 lines of output and send
to your engineer.

---

## What's in this folder

You don't need to touch most of these, but for curiosity:

- `STATE.md` — what step you're on right now
- `DECISIONS.md` — why we chose the settings we did
- `01_scope.md … 07_handoff.md` — full engineering documentation
- `configs/` — training recipes (YAML)
- `datasets/` — your downloaded and cleaned data
- `models/` — your downloaded base models
- `checkpoints/` — output of training
- `eval_results/` — how good the trained model is

If you want to understand any decision, **read DECISIONS.md** — every choice is
explained there.

---

## Glossary (non-engineer-friendly)

- **Base model** — the pretrained LLM you start from (e.g., Llama-3.1-8B)
- **Fine-tuning** — additional training to make the model better at a specific task
- **Dataset** — examples of "the kind of thing I want the model to output"
- **LoRA / QLoRA** — methods that train only a small adapter (≈1% of params), saving 16× memory
- **Epoch** — one pass through the training data
- **Loss** — a number that goes down as the model gets better; if it's NaN, something broke
- **Checkpoint** — a saved snapshot of the model during/after training
- **Eval** — measuring how good the trained model is, on data it never saw during training
