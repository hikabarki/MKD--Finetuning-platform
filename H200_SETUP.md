# H200 Server Setup — MKD Fine-Tuning Platform

Steps to bring the platform up on the 2×H200 server. The Git repo contains the
**code only** — the dataset (~37 GB: JSONL + COCO images) is git-ignored and must
be regenerated or transferred separately (see step 4).

---

## 1. Clone the repo

```bash
git clone git@github.com:mkd-hika/MKD--Finetuning-platform.git ft-platform-refactor
cd ft-platform-refactor
```

## 2. Create the virtualenv (Python 3.10)

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

- The `--extra-index-url` is required for the `torch==2.11.0+cu128` wheels.
  They run on H200 (Hopper, CUDA 12.x). The `nvidia-*` CUDA runtime wheels are
  pulled in automatically by torch.
- If resolution differs from the source machine, fall back to the exact lock:
  `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128`

## 3. (Recommended) Install flash-attn

The trainer falls back to slow eager attention without it. On H200 this is a
large speedup. Install **after** torch is in place:

```bash
pip install flash-attn --no-build-isolation
```

## 4. Get the dataset

The code is on Git; the data is not. Pick one:

### Option A — regenerate on the H200 (clean, reproducible)

```bash
# 4a. Download LLaVA-Instruct-150K conversations and convert to ChatML
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('liuhaotian/LLaVA-Instruct-150K', 'llava_instruct_150k.json', \
  repo_type='dataset', local_dir='/tmp/llava')"
python scripts/convert_llava_to_chatml.py \
  /tmp/llava/llava_instruct_150k.json datasets/llava-instruct-150k

# 4b. Clean / split (writes train.jsonl + val.jsonl)
mv datasets/llava-instruct-150k/train.jsonl datasets/llava-instruct-150k/normalized.jsonl
FT_PROJECT=$PWD python scripts/ft_clean_dataset.py llava-instruct-150k

# 4c. Download COCO 2017 train images (~19 GB) — these are what the JSONL references
mkdir -p datasets/llava-instruct-150k/images
wget -c http://images.cocodataset.org/zips/train2017.zip \
  -O datasets/llava-instruct-150k/images/train2017.zip
unzip -q datasets/llava-instruct-150k/images/train2017.zip \
  -d datasets/llava-instruct-150k/images/

# 4d. Rewrite the `image` field in each row to an ABSOLUTE path under images/train2017/
python - <<'PY'
import json, os
ds = "datasets/llava-instruct-150k"
base = os.path.abspath(f"{ds}/images/train2017")
for split in ("train.jsonl", "val.jsonl", "sample_100.jsonl"):
    p = f"{ds}/{split}"
    if not os.path.exists(p): continue
    rows = [json.loads(l) for l in open(p) if l.strip()]
    for r in rows:
        if r.get("image") and not os.path.isabs(r["image"]):
            r["image"] = os.path.join(base, os.path.basename(r["image"]))
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(split, len(rows), "rows rewritten")
PY
```

### Option B — transfer from the source machine (faster if both are reachable)

```bash
# From the RTX 5090 machine:
rsync -avhP datasets/llava-instruct-150k/ \
  user@h200:/path/to/ft-platform-refactor/datasets/llava-instruct-150k/
```

## 5. Launch

```bash
export FT_PROJECT=$PWD
export ADMIN_PASSWORD=<choose-a-password>   # gates the UI
export API_SECRET=<choose-a-token>          # gates the REST API

python app.py            # Gradio UI  → http://localhost:7860
# and/or
uvicorn api:app --host 0.0.0.0 --port 8000   # REST API
```

---

## H200-specific notes

- **VRAM is huge (2×141 GB).** The `--max-image-tiles 4` cap that was needed to
  fit the 12B VL model on a 32 GB card is unnecessary here. You can raise it back
  toward the model default of 12 for full image resolution:
  `python vl_trainer.py ... --max-image-tiles 12`
- With this much VRAM you can also skip 4-bit and train larger batch sizes, e.g.
  `--batch-size 4` (or higher) and `--quant bf16`.
- For multi-GPU training across both H200s, launch the trainer under `accelerate`
  or `torchrun` (the current `vl_trainer.py` uses a single-GPU `device_map`; a
  multi-GPU launch needs an `accelerate config` pass first).

## Sanity check after setup

```bash
# Confirm images resolve before a long run
head -1 datasets/llava-instruct-150k/train.jsonl | python -c \
  "import sys,json,os; r=json.loads(sys.stdin.readline()); \
   print('image:', r['image'], 'exists:', os.path.exists(r['image']))"
```
