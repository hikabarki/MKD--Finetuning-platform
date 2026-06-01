# 02 — Architecture (Provisional)

> **Status:** provisional. Final version is written after Phase 1 questions
> are answered. The defaults below assume the **single H100 80 GB, ≤13B model,
> 1 user, SFT+DPO, local-disk** profile, which is the modal MKD setup.

## Pipeline overview

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│  ft add-dataset │──▶│  ft clean-dataset │──▶│   ft add-model  │
│  (download +    │    │  (dedup, filter,  │    │  (HF download, │
│   normalize)    │    │   PII scrub)      │    │   verify tmpl) │
└─────────────────┘    └──────────────────┘    └────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│  ft publish     │◀──│   ft export       │◀──│  ft configure   │
│  (HF + GitHub)  │    │  (merge, quant)   │    │  (gen YAML)    │
└─────────────────┘    └──────────────────┘    └────────────────┘
                              ▲                          │
                              │                          ▼
                       ┌──────────────────┐    ┌────────────────┐
                       │     ft eval      │◀──│   ft train      │
                       │  (lm-eval, A/B)  │    │  (axolotl)     │
                       └──────────────────┘    └────────────────┘
                                                        ▲
                                                        │
                                               ┌────────────────┐
                                               │  ft smoke-test │
                                               │  (50 steps)    │
                                               └────────────────┘
```

Every box ends by printing the next command. The arrows are not just data flow —
they are literal printed strings in stdout.

## Component choices (provisional)

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Dataset download | `datasets` + `huggingface_hub` + `requests` | Standard, handles HF, http, local |
| Format normalization | Custom converter → ChatML JSONL | Single canonical format upstream of every trainer |
| Dedup | MinHash LSH via `datasketch` | O(n) approximate dedup that scales to 10M rows on one machine |
| Decontamination | 13-gram overlap vs eval sets | Standard practice (Llama, Qwen technical reports) |
| PII scan | Microsoft `presidio` | Mature, configurable, on-CPU |
| Tokenizer fidelity | `tokenizer.apply_chat_template()` round-trip test | Catches the #1 silent killer of FT runs |
| Training backend (SFT/DPO) | **Axolotl** | YAML-first, FSDP+DeepSpeed, broad model support, active maintenance |
| Fast-path single-GPU | **Unsloth** | 2× speed, 60% less VRAM for LoRA, used when DP=1 detected |
| Eval | `lm-evaluation-harness` + `MT-Bench` | Standard capability eval + judge eval |
| Serving merge/quant | `peft` (merge) + `autoawq` or `llama.cpp` (quantize) | AWQ for vLLM, GGUF for llama.cpp |
| Inference benchmark | `vllm` (1k prompts, p50/p99 latency) | Production-like load |
| Publish | `huggingface_hub.upload_folder` → `mkd-hika/<repo>` | Per Hika identity |

## Memory budget — illustrative (8B model, QLoRA, H100 80GB)

| Item | Bytes/param | Total (8B) |
|------|-------------|-----------|
| Base weights (4-bit NF4) | 0.5 | 4.0 GB |
| LoRA adapters (rank 32, bf16) | ~0.5% of params × 2 | 80 MB |
| Optimizer (8-bit AdamW for LoRA only) | 4 / adapter param | 320 MB |
| Gradients (bf16, LoRA only) | 2 / adapter param | 160 MB |
| Activations (seq=4096, batch=4, grad-ckpt) | — | ~12 GB |
| KV cache (training) | — | ~3 GB |
| **Total** | | **~20 GB** — fits with headroom on 80GB |

For a **full SFT** of the same 8B model: ~96 GB → spills to multi-GPU or
requires DeepSpeed ZeRO-3 / FSDP.

## Parallelism strategy (provisional)

| Hardware | Strategy |
|----------|----------|
| Single 24GB GPU | QLoRA, Unsloth, grad-ckpt on, micro-batch 1, grad-accum 16 |
| Single 80GB GPU | QLoRA or LoRA, Axolotl, micro-batch 4–8, grad-accum 4 |
| 4–8× H100 | FSDP full-shard, bf16, optional FP8 via TransformerEngine |
| >8× / multi-node | DeepSpeed ZeRO-3 + offload, FSDP for ≤70B, Megatron-PP for ≥70B |

## Precision policy

- **Default:** bf16 mixed precision (master weights bf16, no fp32 copy needed
  for LoRA; fp32 master only for full SFT).
- **FP8 (H100/H200):** opt-in via `precision: fp8` in recipe YAML; requires
  TransformerEngine; flag as experimental until we have a regression eval baseline.
- **NF4 (QLoRA):** the default base-weights quantization for adapter training.

## Failure-mode catalog (top 5 to preempt)

1. **Wrong chat template applied** → silent 30-50% eval drop.
   *Fix:* every `ft add-model` runs a round-trip rendering test and refuses if
   the model card lacks a documented template.
2. **OOM at step ~N** (activations grow with seq len of the longest packed batch).
   *Fix:* `ft smoke-test` runs with the 99th-percentile-length batch on purpose.
3. **NCCL hang on multi-GPU** at first all-reduce.
   *Fix:* default `NCCL_DEBUG=INFO` and a 60s `NCCL_TIMEOUT_S` in the env file.
4. **Tokenizer mismatch after resume** (added special tokens not re-embedded).
   *Fix:* the manifest checks `tokenizer.json` hash on every checkpoint load.
5. **Eval-set contamination** discovered after publishing.
   *Fix:* `ft clean-dataset` writes `eval_results/<ds>_contamination.json`; the
   `ft publish` script refuses to push if contamination ratio > 0.5%.
