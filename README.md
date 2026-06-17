# MKD Fine-Tuning Platform — Multimodal

A production-grade fine-tuning platform for NVIDIA GPUs supporting both **text** and **vision-language (VL)** models. Built with a Gradio UI for rapid prototyping and a clean CLI backend designed for frontend integration.

---

## Features

- **Multimodal training** — fine-tune vision-language models (NVIDIA Nemotron-Nano-12B-VL) on image+text datasets
- **Text fine-tuning** — LoRA/QLoRA fine-tuning via Axolotl on any JSONL/CSV instruction dataset
- **General User Mode** — Quick Start and Custom Mix flows for non-technical users
- **Expert Mode** — full control over model, dataset, quantization, learning rate, epochs, and evaluation
- **Model Evaluation Dashboard** — training loss, perplexity, before/after comparison charts for every run
- **Multimodal Chat** — test your fine-tuned model with image + text inputs directly in the UI
- **Export & Publish** — merged checkpoint export with LoRA adapter merge
- **Automatic eval registration** — every completed training run appears in the Evaluation Dashboard automatically

---

## Stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio 6.x |
| VL Training | Custom `vl_trainer.py` (HuggingFace Transformers + PEFT) |
| Text Training | [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) |
| Quantization | bitsandbytes (4-bit QLoRA), BF16 |
| Inference | vLLM (text), direct HF (VL) |
| GPU | NVIDIA RTX 5090 / B200 (CUDA 12.8, PyTorch 2.11) |

---

## Supported Models

- **NVIDIA Nemotron-Nano-12B-VL BF16** (vision-language, default)
- **NVIDIA Nemotron-3-Nano-4B BF16** (text, fallback)
- Any HuggingFace-compatible causal LM

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt   # or: bash install.sh

# Launch the UI
python app.py

# With public share link
python app.py --share

# CLI usage
ft configure --model=<model> --dataset=<dataset>
ft train configs/<run>.yml
ft eval checkpoints/<run>
ft export checkpoints/<run> --merge
```

---

## Project Structure

```
ft-platform-multimodal/
├── app.py                  # Gradio UI (General User + Expert Mode)
├── vl_trainer.py           # Vision-language fine-tuning trainer
├── api.py                  # REST API layer
├── setup.py                # Package setup
├── install.sh              # Dependency installer
├── scripts/
│   ├── ft                  # CLI entry point
│   ├── ft_train.py         # Training orchestration
│   ├── ft_eval.py          # Evaluation (lm-eval-harness)
│   ├── ft_configure.py     # Recipe/config generation
│   ├── ft_export.py        # Checkpoint export + LoRA merge
│   ├── ft_add_dataset.py   # Dataset ingestion
│   ├── ft_clean_dataset.py # Dataset cleaning + PII scrub
│   └── ft_common.py        # Shared utilities
├── configs/                # Generated training recipes (gitignored: run_*)
├── data_cards/             # Dataset metadata
└── eval_results/           # Per-run evaluation results
```

---

## API

The Gradio UI exposes a full REST API automatically. With the server running:

```
http://localhost:7860/?view=api
```

See `API_SPEC.md` and `FRONTEND_CONTRACT.md` for the frontend integration contract.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 24 GB VRAM | 32 GB+ (RTX 5090 / B200) |
| CUDA | 12.8 | 12.8 |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB | 500 GB |

> **Note:** NVFP4 quantization is inference-only. Training uses BF16 or 4-bit QLoRA.

---

## Evaluation Dashboard

Every training run (Quick Start, Custom Mix, Expert) is automatically registered in the Evaluation Dashboard after completion. The dashboard shows:

- Training loss curve
- Perplexity over steps
- Before vs After comparison (loss + perplexity)
- Summary table with % improvement

---

## License

Internal use — MKD © 2026
