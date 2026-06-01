# 06 — Evaluation & Decontamination Audit

## Philosophy

Held-out loss is **necessary but insufficient**. A model can have lower val loss
and worse downstream capability. Every fine-tune is evaluated on:

1. **Held-out loss** (sanity)
2. **Capability evals** (does it know things)
3. **Instruction-following** (does it do what you ask)
4. **Regression** (did we break the base model)
5. **Contamination audit** (are the eval numbers trustworthy)

## Default eval suite

| Layer | Eval | Why |
|-------|------|-----|
| Loss | Per-step val loss, plotted | Trend, plateau detection |
| Capability | `lm-eval-harness`: MMLU, ARC-c, HellaSwag, TruthfulQA, GSM8K | Broad coverage in ~30 min |
| Code (if relevant) | HumanEval, MBPP (pass@1) | Code regression |
| Instruction-following | IFEval, MT-Bench (GPT-4 judge), AlpacaEval 2 | Chat quality |
| Korean (MKD-relevant) | KMMLU, HAE-RAE Bench, KoBEST | Domestic capability |
| Safety | XSTest, ToxiGen | Doesn't get worse than base |
| Regression | All of above run on the **base model** too; report delta | The only meaningful comparison |

The eval suite is configurable per project; the default is a JSON file:
`configs/eval_default.json`. `ft eval` reads it, runs every listed eval, writes
to `eval_results/<run-id>_<eval-name>.json` (immutable, timestamped).

## Decontamination check

For every eval set, `ft eval` computes a 13-gram overlap report against the
training data:

```json
// eval_results/<run-id>_contamination.json
{
  "mmlu": {"contaminated_rows": 12, "total": 14042, "ratio": 0.00085, "verdict": "ok"},
  "gsm8k": {"contaminated_rows": 0, "total": 1319, "ratio": 0.0, "verdict": "ok"},
  "humaneval": {"contaminated_rows": 47, "total": 164, "ratio": 0.286, "verdict": "FAIL"}
}
```

`ft publish` refuses to push if any eval has ratio > 0.005 (0.5%) without
`--acknowledge-contamination`.

## A/B vs base model

The most important number: **win-rate against the base model on a held-out
prompt set**.

```bash
ft eval-vs-base checkpoints/run_2026-05-26 \
    --base meta-llama/Llama-3.1-8B-Instruct \
    --prompts datasets/alpaca/eval_prompts_500.jsonl \
    --judge gpt-4o
```

Output: `eval_results/<run-id>_vs_base.json`

```json
{
  "n_prompts": 500,
  "win_rate": 0.612,           // 95% CI: [0.575, 0.649]
  "tie_rate": 0.124,
  "lose_rate": 0.264,
  "wilcoxon_p": 0.0003,
  "categories": {
    "reasoning": 0.58,
    "coding": 0.71,
    "knowledge": 0.49,
    "korean": 0.81           // Where the fine-tune actually helped
  }
}
```

Honest interpretation guide is printed by `ft eval-vs-base`:
- Win rate < 0.55 with overlapping CIs → **no meaningful improvement.** Reconsider
  data, LR, or epochs before publishing.
- Improvement only in one narrow category but regression elsewhere → flag the
  user; this might be desired (domain adapt) or a bug (catastrophic forgetting).

## What gets archived

`eval_results/` is immutable. Every run produces:

```
eval_results/run_2026-05-26/
├── lm_eval_harness.json
├── mt_bench.json
├── alpaca_eval_2.json
├── kmmlu.json
├── contamination.json
├── vs_base.json
├── env_snapshot.txt        # pip freeze, nvidia-smi, OS, driver
└── summary.md              # human-readable digest
```

`summary.md` is what gets linked from the MODEL_CARD.md on publish.
