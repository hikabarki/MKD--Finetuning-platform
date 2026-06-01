# 07 — Deployment & Handoff

## Outputs of `ft export`

```bash
ft export checkpoints/run_2026-05-26 --merge --quantize awq
```

Produces:

```
exports/run_2026-05-26/
├── merged/                  # LoRA merged into base, bf16
├── awq/                     # 4-bit AWQ for vLLM
├── gguf/                    # Q4_K_M for llama.cpp (optional, --gguf flag)
├── MODEL_CARD.md            # auto-generated, see template below
├── tokenizer/               # copy of tokenizer with any added tokens
└── inference_test.json      # output of a 20-prompt smoke test
```

## Quantization choices

| Format | Use case | Loss vs bf16 (typical) |
|--------|----------|------------------------|
| bf16 (no quant) | Research, max quality | 0 |
| AWQ 4-bit | vLLM / SGLang serving | < 1% on MMLU |
| GPTQ 4-bit | Older GPUs without bf16 | ~1-2% |
| GGUF Q4_K_M | llama.cpp / on-device | ~1% |
| GGUF Q5_K_M | llama.cpp / better quality | ~0.3% |
| FP8 (TE) | H100/H200 serving | ~0% but model-specific |

Default = AWQ. Other formats opt-in via flags.

## Inference benchmark

`ft export` always runs a 100-prompt benchmark on the chosen quant format using
vLLM and writes:

```
exports/run_<id>/benchmark.json
{
  "model": "exports/run_2026-05-26/awq",
  "engine": "vllm-0.7.3",
  "gpu": "H100-80GB-PCIe",
  "concurrency": 32,
  "input_tokens_avg": 512,
  "output_tokens_avg": 256,
  "throughput_tok_s": 4830,
  "latency_p50_ms": 482,
  "latency_p99_ms": 1240
}
```

## MODEL_CARD.md template

Auto-generated. The user reviews and edits before `ft publish`:

```markdown
# {repo-name}

Fine-tuned from [{base-model}]({base-url}) on [{dataset}]({dataset-url}).

## Intended use
{filled by user — required to be non-empty before publish}

## Training details
- Method: {QLoRA / LoRA / full SFT}
- Hardware: {1× H100 80GB}
- Duration: {6h 13m}
- Dataset rows: {49,606 train, 502 val}
- Final val loss: {0.83}

## Evaluation
| Eval | Base | This model | Δ |
|------|------|-----------|---|
| MMLU | 0.68 | 0.69 | +0.01 |
| MT-Bench | 7.8 | 8.2 | +0.4 |
| Win rate vs base | — | 0.61 | — |

Full report: [eval_results/run_2026-05-26/summary.md](...)

## Bias, risks, limitations
{filled by user}

## License
Inherits {base-model-license}. See LICENSE.

## Citation
{filled by user, optional}
```

## Publish — `ft publish`

```bash
ft publish exports/run_2026-05-26 \
    --hf-repo mkd-hika/llama-3.1-8b-alpaca-ko \
    --github-repo mkd-hika/llama-3.1-8b-alpaca-ko
```

What it does (and what it refuses to do):

1. **Refuses if:**
   - `MODEL_CARD.md` has any `{filled by user}` placeholders left
   - Contamination ratio > 0.5% on any eval (unless `--acknowledge-contamination`)
   - Base model license is research-only and `--accept-base-license` was not passed
   - `git status` in the project folder is dirty (uncommitted ADRs etc.)

2. **HuggingFace push:**
   ```bash
   huggingface-cli upload mkd-hika/llama-3.1-8b-alpaca-ko exports/run_<id>/awq . \
       --repo-type=model \
       --commit-message="Initial release"
   ```

3. **GitHub push** (training code + configs, not weights):
   ```bash
   git init && git add . && git commit -m "Initial release"
   git branch -M main
   git remote add origin https://github.com/mkd-hika/llama-3.1-8b-alpaca-ko.git
   git push -u origin main
   ```

4. **Prints final summary:**
   ```
   🎉 Published.
      Weights:   https://huggingface.co/mkd-hika/llama-3.1-8b-alpaca-ko
      Code:      https://github.com/mkd-hika/llama-3.1-8b-alpaca-ko
      Model card: https://huggingface.co/mkd-hika/llama-3.1-8b-alpaca-ko/blob/main/README.md
   ```

## After publish — pointers

- Monitor download / use stats on the HF repo page.
- If a regression is reported, do **not** force-push over the existing version.
  Train a v2 in a new run folder; publish as `<repo-name>-v2`. Immutability of
  released artifacts is a core MKD policy.
