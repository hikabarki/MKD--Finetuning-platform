# Recommended Datasets for MKD Fine-Tuning Platform

Based on the existing datasets (Alpaca, Alpaca-tiny, Guanaco-1k) and the project's
Korean-focused deployment (`mkd-hika` W&B entity), the following datasets are recommended.

---

## 1. Korean Instruction — `HAERAE-HUB/KMMLU`
| Field | Value |
|---|---|
| **HuggingFace ID** | `HAERAE-HUB/KMMLU` |
| **Size** | ~35 k rows across 45 Korean subject domains |
| **Format** | Multiple-choice QA (question + options + answer) → converted to Alpaca-style |
| **Why useful** | Comprehensive Korean factual + reasoning coverage; aligns with MKD's Korean focus. KMMLU is the Korean equivalent of MMLU. |
| **Download** | `ft add-dataset HAERAE-HUB/KMMLU --name kmmlu` |

---

## 2. Korean General Instruction — `nayohan/llama3-instrction-ko-dataset`
| Field | Value |
|---|---|
| **HuggingFace ID** | `nayohan/llama3-instrction-ko-dataset` |
| **Size** | ~50 k conversational instruction rows |
| **Format** | ShareGPT / ChatML — auto-detected by `ft add-dataset` |
| **Why useful** | Korean conversational instruction-following tuned for LLaMA-3 style chat templates — compatible with Gemma 4 (also LLaMA-3 based). |
| **Download** | `ft add-dataset nayohan/llama3-instrction-ko-dataset --name ko-instruct` |

---

## 3. Multimodal VQA — `HuggingFaceM4/NoCaps`
| Field | Value |
|---|---|
| **HuggingFace ID** | `HuggingFaceM4/NoCaps` |
| **Size** | ~15 k image-caption pairs (validation split only; test is held-out) |
| **Format** | Image + caption → converted to image-caption ChatML by `ft add-dataset` |
| **Why useful** | High-quality out-of-domain captioning benchmark images; small enough for quick VQA testing with Gemma 4 12B. Images are COCO-format with diverse subjects. |
| **Download** | `ft add-dataset HuggingFaceM4/NoCaps --name nocaps-vqa` |

---

## 4. Multilingual Instruction — `CohereForAI/aya_dataset`
| Field | Value |
|---|---|
| **HuggingFace ID** | `CohereForAI/aya_dataset` |
| **Size** | ~202 k rows across 65 languages including Korean |
| **Format** | Alpaca-style (`inputs` + `targets`) — auto-detected |
| **Why useful** | Multilingual instruction data with Korean coverage; good for general Korean + English bilingual fine-tuning. Filter to Korean with `ft clean-dataset aya --lang ko,en`. |
| **Download** | `ft add-dataset CohereForAI/aya_dataset --name aya` |

---

## 5. Multimodal VQA (Korean) — `snunlp/VQA_Korean`
| Field | Value |
|---|---|
| **HuggingFace ID** | `snunlp/VQA_Korean` |
| **Size** | ~30 k Korean VQA pairs |
| **Format** | VQA (image + question + answer) — auto-detected |
| **Why useful** | Korean-language visual question answering; combines the project's two main focus areas (Korean language + multimodal). Use for fine-tuning and evaluation of the VL model. |
| **Download** | `ft add-dataset snunlp/VQA_Korean --name ko-vqa` |

---

## End-to-end example: add → clean → train (Korean VQA)

```bash
# 1. Add Korean VQA dataset
ft add-dataset snunlp/VQA_Korean --name ko-vqa

# 2. Clean (keep Korean + English, 1% val split)
ft clean-dataset ko-vqa --lang ko,en --val-ratio 0.01

# 3. Check 10 rows before training
head -3 datasets/ko-vqa/sample_100.jsonl

# 4. Fine-tune the VL model
ft configure --model gemma-4-12b \
             --dataset ko-vqa --mode vl --quant bf16

# 5. Train (smoke test first)
ft smoke-test configs/run_<timestamp>.yaml
ft train configs/run_<timestamp>.yaml

# 6. Evaluate (runs VQA accuracy + contamination check)
ft eval checkpoints/vl_<timestamp>

# 7. Export + publish
ft export checkpoints/vl_<timestamp> --merge
ft publish exports/run_<timestamp>/merged --hf-repo mkd-hika/gemma-ko-vqa
```
