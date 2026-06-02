#!/usr/bin/env python3
"""
ft_configure.py — generate a runnable training recipe YAML from model + dataset.

Usage:
    python ft_configure.py --model <model-name-or-path> --dataset <dataset-name>
                           [--quantize 4bit|8bit|none] [--lr <float>]

Steps:
    1. Load MODEL_INFO.json for the model, data_card for the dataset.
    2. Detect GPU(s) with nvidia-smi.
    3. Pick sane defaults based on detected hardware and model size.
    4. Write configs/run_<timestamp>.yaml (Axolotl-compatible).
    5. Print: ft smoke-test configs/run_<ts>.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (
    project_root, info, ok, warn, err, next_step,
    update_state,
)


def detect_gpu() -> dict:
    """Return first GPU's name and memory (GB), or CPU fallback."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().splitlines()
        if out:
            name, mem_mib = out[0].split(", ")
            mem_gb = int(mem_mib) / 1024
            count = len(out)
            return {"name": name.strip(), "mem_gb": round(mem_gb, 1), "count": count}
    except Exception:
        pass
    return {"name": "CPU", "mem_gb": 0, "count": 0}


def pick_profile(gpu: dict, params_b: float, quantize: str) -> dict:
    """Return training hyperparameter defaults for the detected hardware."""
    mem = gpu["mem_gb"]

    if mem == 0:
        return {"adapter": "qlora", "load_in_4bit": True, "micro_batch_size": 1,
                "gradient_accumulation_steps": 16, "optimizer": "paged_adamw_8bit",
                "flash_attention": False, "note": "CPU mode — only for testing"}

    if quantize == "none":
        adapter = "lora"
        load_4 = False
    else:
        adapter = "qlora"
        load_4 = True

    if mem >= 70:
        micro = 8; grad_acc = 4; opt = "paged_adamw_8bit"
    elif mem >= 40:
        micro = 4; grad_acc = 8; opt = "paged_adamw_8bit"
    else:
        micro = 1; grad_acc = 16; opt = "adamw_torch"
        if params_b > 13:
            warn(f"Model is {params_b:.0f}B but GPU has only {mem:.0f} GB — forcing 4-bit QLoRA")
            adapter = "qlora"; load_4 = True

    return {
        "adapter": adapter,
        "load_in_4bit": load_4,
        "micro_batch_size": micro,
        "gradient_accumulation_steps": grad_acc,
        "optimizer": opt,
        "flash_attention": True,
    }


def find_model_dir(root: Path, model_name: str) -> Path | None:
    """Match model_name against models/ subdirectory names (fuzzy)."""
    models_dir = root / "models"
    if not models_dir.exists():
        return None
    needle = model_name.lower().replace("-", "").replace("_", "").replace("/", "")
    for d in models_dir.iterdir():
        hay = d.name.lower().replace("-", "").replace("_", "").replace("__", "")
        if needle in hay or hay.endswith(needle):
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model name as saved under models/")
    ap.add_argument("--dataset", required=True, help="Dataset name as saved under datasets/")
    ap.add_argument("--quantize", default="4bit", choices=["4bit", "8bit", "none"],
                    help="Base-weights quantization (default: 4bit QLoRA)")
    ap.add_argument("--lr", type=float, default=None, help="Override learning rate")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--wandb-project", default="", help="W&B project name (empty = disabled)")
    ap.add_argument("--wandb-entity",  default="", help="W&B entity/username (empty = disabled)")
    args = ap.parse_args()

    root = project_root()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M")

    # -- Dataset card --
    card_path = root / "data_cards" / f"{args.dataset}.json"
    if not card_path.exists():
        err(f"No data card found for '{args.dataset}'. Run `ft clean-dataset {args.dataset}` first.")
        sys.exit(2)
    card = json.loads(card_path.read_text())
    train_path = f"datasets/{args.dataset}/train.jsonl"
    if not (root / train_path).exists():
        err(f"Training data not found at {train_path}. Run `ft clean-dataset {args.dataset}` first.")
        sys.exit(2)

    # -- Model info --
    model_dir = find_model_dir(root, args.model)
    model_info_path = model_dir / "MODEL_INFO.json" if model_dir else None
    if model_info_path and model_info_path.exists():
        minfo = json.loads(model_info_path.read_text())
        model_id = minfo["model_id"]
        params_b = minfo.get("params_billion", 8.0)
        local_model = str(model_dir.relative_to(root))
        info(f"Found model: {model_id}  ({params_b:.1f}B params)")
    else:
        warn(f"No MODEL_INFO.json found for '{args.model}' — using name as HF ID")
        model_id = args.model
        params_b = 8.0
        local_model = args.model

    # -- GPU detection --
    gpu = detect_gpu()
    info(f"GPU: {gpu['name']}  {gpu['mem_gb']:.0f} GB  ×{gpu['count']}")

    profile = pick_profile(gpu, params_b, args.quantize)
    lr = args.lr or 2e-4

    # -- Detect chat template name from model id --
    mid_lower = model_id.lower()
    if "llama-3" in mid_lower or "llama3" in mid_lower:
        chat_template = "llama3"
    elif "qwen2" in mid_lower or "qwen-2" in mid_lower:
        chat_template = "chatml"
    elif "mistral" in mid_lower:
        chat_template = "mistral"
    elif "gemma" in mid_lower:
        chat_template = "gemma"
    else:
        chat_template = "chatml"
        warn(f"Could not auto-detect chat_template for '{model_id}' — defaulting to chatml. "
             f"Edit configs/run_{ts}.yaml if needed.")

    run_name = f"{args.model.split('/')[-1].lower()}-{args.dataset}-{ts}"
    out_yaml = root / "configs" / f"run_{ts}.yaml"
    out_yaml.parent.mkdir(exist_ok=True)

    yaml_content = f"""\
# =============================================================================
# Training recipe — generated by `ft configure` at {ts} UTC
# Model:   {model_id}
# Dataset: {args.dataset}
# GPU:     {gpu['name']} {gpu['mem_gb']:.0f} GB ×{gpu['count']}
# =============================================================================

# --- Model -------------------------------------------------------------------
base_model: {local_model}
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
trust_remote_code: false

# --- Dataset -----------------------------------------------------------------
datasets:
  - path: {train_path}
    type: chat_template
    field_messages: messages
    message_field_role: role
    message_field_content: content

chat_template: {chat_template}
val_set_size: 0.01
sequence_len: {args.seq_len}
sample_packing: false
pad_to_sequence_len: true

# --- Method ------------------------------------------------------------------
adapter: {profile['adapter']}
load_in_4bit: {str(profile['load_in_4bit']).lower()}
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_use_double_quant: true

lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# --- Optimizer & schedule ----------------------------------------------------
optimizer: {profile['optimizer']}
learning_rate: {lr}
lr_scheduler: cosine
warmup_ratio: 0.03
weight_decay: 0.0
max_grad_norm: 1.0
adam_beta1: 0.9
adam_beta2: 0.95
adam_epsilon: 1.0e-8

# --- Run ---------------------------------------------------------------------
num_epochs: {args.epochs}
micro_batch_size: {profile['micro_batch_size']}
gradient_accumulation_steps: {profile['gradient_accumulation_steps']}
gradient_checkpointing: true
bf16: true
fp16: false
tf32: true
flash_attention: {str(profile['flash_attention']).lower()}

# --- Eval & checkpoint -------------------------------------------------------
eval_steps: 100
save_steps: 200
save_total_limit: 3
logging_steps: 10
output_dir: checkpoints/run_{ts}

# --- Logging -----------------------------------------------------------------
wandb_project: "{args.wandb_project}"
wandb_entity: "{args.wandb_entity}"
wandb_name: "{run_name if args.wandb_project else ''}"
wandb_log_model: "false"

# --- Stability ---------------------------------------------------------------
seed: 42
deterministic: false
loss_watchdog_threshold: 5.0
loss_watchdog_patience: 3
"""

    out_yaml.write_text(yaml_content)
    ok(f"Wrote {out_yaml}")

    update_state(
        phase="4 / Recipe generated",
        status=f"✓ Recipe written: configs/run_{ts}.yaml",
        next_action=f"Smoke-test the recipe: `ft smoke-test configs/run_{ts}.yaml`",
        done=[f"Generated recipe for {model_id} × {args.dataset}"],
    )
    next_step(
        commands=[f"ft smoke-test configs/run_{ts}.yaml"],
        note="50-step dry run — confirms memory fit, loss shape, checkpoint save/load. Always run before full training.",
    )


if __name__ == "__main__":
    main()
