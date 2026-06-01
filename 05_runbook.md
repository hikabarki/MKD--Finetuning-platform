# 05 — Runbook

> "Smoke-test before full run" is non-negotiable. It is 30 minutes that saves 30 hours.

## Pre-flight checklist (script: `ft smoke-test`)

The smoke test runs **before every full training run**, no exceptions. It performs:

1. **Disk space check.** Need ≥3× model size free in `checkpoints/`.
2. **GPU detection.** `nvidia-smi` parse → confirm count, memory, driver ≥ 555 for H100.
3. **NCCL ping** (multi-GPU only). Tiny all-reduce; abort if RTT > 1 ms.
4. **Tokenizer round-trip.** Render a sample with chat template, tokenize, detokenize,
   diff against the original message-by-message. Any mismatch = abort.
5. **Dataloader sanity.** Pull 10 batches; check labels are masked outside assistant
   turns (this is the single most common data bug).
6. **50-step training run** with the *real* model and *real* data:
   - Confirms it fits in memory.
   - Confirms loss decreases (or at least stays bounded).
   - Confirms grad_norm is in [0.1, 10].
   - Confirms checkpoint save+reload works (write at step 50, reload, check state-dict equality).
7. **Report.** Prints token/sec, MFU%, peak VRAM, ETA for full run, $ estimate.

## Full-run procedure

```bash
# 1. Generate recipe (done once per project)
ft configure --model llama-3.1-8b --dataset alpaca

# 2. Smoke test (always)
ft smoke-test configs/run_2026-05-26.yaml

# 3. Read the smoke-test report. If it says "✓ ready to train", proceed.
#    If it says "⚠️ ..." — fix and re-smoke. Don't push through warnings.

# 4. Full training (in a screen / tmux session — this is hours-to-days)
tmux new -s train
ft train configs/run_2026-05-26.yaml

# 5. Monitor (separate terminal)
ft watch run_2026-05-26
# → prints rolling loss, grad_norm, tok/s, ETA, sample generation every 200 steps
```

## Abort criteria (auto-enforced by `ft train`)

| Signal | Threshold | Action |
|--------|-----------|--------|
| Loss is NaN/Inf | 1 occurrence | Abort, save crash dump |
| grad_norm | > 10 for 5 consecutive steps | Abort, save crash dump |
| Loss spike | > 5× rolling median for 3 steps | Abort, save crash dump |
| tok/s | drops > 30% from minute-1 baseline | Warn, continue, log to `logs/` |
| MFU | < 20% at step 50 | Warn — likely a config bug |
| Disk free | < 2× checkpoint size | Pause, alert, wait for human |

A "crash dump" = the last 100 steps of logs + the most recent checkpoint manifest
+ a `post_mortem.md` template prefilled with the abort reason and the env hash.

## Resume procedure

```bash
ft resume checkpoints/run_2026-05-26/step_1200
```

`ft resume` verifies:
- Tokenizer SHA matches
- Dataset SHA matches (refuses if data changed mid-run — common silent killer)
- Optimizer state present
- Re-seeds RNG from `<ckpt>/rng_state.pt` to keep determinism across resumes

If any check fails, it prints a remediation message and refuses to resume.

## What "done" looks like

Training is "done" when **all** of these are true:
- ✅ Final epoch completed without aborts
- ✅ Val loss curve has plateaued (not strictly decreasing for ≥ 200 steps)
- ✅ Sample generations at step N look qualitatively sensible (eyeball check)
- ✅ `eval_results/run_<id>_final.json` exists with all eval metrics
- ✅ Checkpoint passes a load-and-generate test

The script `ft train` prints exactly this checklist when it finishes, and tells
the user the next command:

```
✓ Training complete. 12,400 steps, final val loss 0.83, MFU 47%.
  Total cost estimate: $148 (H100, 6h13m at $24/h).

  Next: ft eval checkpoints/run_2026-05-26 --base meta-llama/Llama-3.1-8B-Instruct
```
