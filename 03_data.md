# 03 — Data Pipeline

## Canonical format

All datasets are normalized into **ChatML JSONL** before training. One example
per line:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is the capital of France?"},
  {"role": "assistant", "content": "Paris."}
]}
```

For DPO/ORPO:

```json
{"prompt": [...messages up to last user turn...],
 "chosen": {"role": "assistant", "content": "..."},
 "rejected": {"role": "assistant", "content": "..."}}
```

Why ChatML upstream of every trainer:
1. It maps cleanly to every modern chat template (Llama-3, Qwen2.5, Mistral, Gemma).
2. `tokenizer.apply_chat_template()` does the per-model rendering at training time,
   so the on-disk data is model-agnostic.
3. DPO/ORPO data is a superset of SFT data with two assistant choices.

## Supported input formats

The `ft add-dataset` script auto-detects and converts:

| Input shape | Detection | Conversion |
|-------------|-----------|------------|
| HF dataset with `messages` column | column name | passthrough |
| Alpaca (`instruction`, `input`, `output`) | column names | `system` from instruction, user from input, assistant from output |
| ShareGPT (`conversations` w/ `from`/`value`) | column names | role remap: `human`→user, `gpt`→assistant |
| OpenAI (`prompt`, `completion`) | column names | wrap into 2-turn |
| Raw text (`text` column, single column) | only 1 col | error: "ambiguous, please specify" |
| DPO (`prompt`, `chosen`, `rejected`) | column names | passthrough |
| CSV / Parquet / JSONL | extension | infer columns then route above |

## Cleaning steps (in order)

Each step is logged with row counts in/out into `data_cards/<name>.json`.

1. **Schema validation.** Drop rows missing required fields. Log count.
2. **Length filtering.** Drop rows where any single message > `max_msg_tokens`
   (default 8192) or total conversation > `max_seq_tokens` (default 32768).
   Tokenization done with the *target model's* tokenizer.
3. **Language filter (optional).** `fasttext` lid.176 model; default "keep en+ko".
4. **PII scan.** Microsoft Presidio; default action **mask** email/phone/SSN,
   **drop** rows with credit cards. Configurable in recipe.
5. **Deduplication.** MinHash LSH (`num_perm=128`, threshold 0.85). Reports both
   exact and near-duplicates removed.
6. **Decontamination.** 13-gram overlap test against every eval set in
   `eval_results/`. Rows with > 50% 13-gram overlap with any eval row are dropped.
7. **Quality filter (optional, off by default).** A small classifier
   (`HuggingFaceH4/fineweb-edu-classifier`-style) scores each row 0–5; keeps
   rows ≥ 2. Off by default because it's slow and domain-specific.
8. **Train/val split.** Default 99/1, seeded. Val set is never deduped against
   train (we want representative val).
9. **Packing.** Off by default (preserves loss-mask boundaries). When on, uses
   "best-fit decreasing" with EOS separators; can be turned on via recipe.

## Data card schema

`data_cards/<name>.json`:

```json
{
  "name": "alpaca-cleaned-v1",
  "source": "https://huggingface.co/datasets/yahma/alpaca-cleaned",
  "license": "CC-BY-4.0",
  "downloaded_at": "2026-05-26T10:00:00Z",
  "format": "alpaca",
  "rows_raw": 52002,
  "rows_after_schema": 52002,
  "rows_after_length": 51840,
  "rows_after_pii": 51838,
  "rows_after_dedup": 50112,
  "rows_after_decontam": 50108,
  "train_rows": 49606,
  "val_rows": 502,
  "tokenizer_used": "meta-llama/Llama-3.1-8B-Instruct",
  "avg_seq_tokens": 287,
  "p99_seq_tokens": 1843,
  "sha256_train": "...",
  "sha256_val": "...",
  "pii_report": "eval_results/alpaca-cleaned-v1_pii_report.json",
  "contamination_report": "eval_results/alpaca-cleaned-v1_contamination.json"
}
```

## Manual-review sample

`ft clean-dataset` always emits `datasets/<name>/sample_100.jsonl` — a random
100-row sample for a human to eyeball before training. The script's printed
"next step" is:

```
✓ Cleaning done. Please review datasets/alpaca/sample_100.jsonl by eye.
  If it looks correct, run:    ft add-model <hf-model-id>
  If it looks wrong, run:      ft inspect-dataset alpaca   (interactive view)
```

This is the only place we ask a non-engineer to do something non-mechanical,
and it's deliberate — a 30-second eyeball check catches catastrophic format
errors that no automated test will.
