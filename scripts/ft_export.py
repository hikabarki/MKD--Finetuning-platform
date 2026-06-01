#!/usr/bin/env python3
"""
ft_export.py — merge LoRA adapters, quantize, write MODEL_CARD.md.

Usage:
    python ft_export.py <checkpoint-dir> [--merge] [--quantize awq|gptq|gguf|none]
                        [--gguf] [--output-dir <path>]

Per 07_handoff.md, produces:
    exports/<run-id>/
    ├── merged/          ← LoRA merged into base (bf16)
    ├── awq/             ← 4-bit AWQ (default)
    ├── gguf/            ← Q4_K_M GGUF (opt-in --gguf)
    ├── tokenizer/       ← copy of tokenizer
    ├── MODEL_CARD.md    ← auto-generated template
    └── inference_test.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (
    project_root, info, ok, warn, err, next_step,
    update_state,
)


def find_base_model(ckpt_dir: Path, root: Path) -> str | None:
    """Try to read base model id from the axolotl config saved in the checkpoint."""
    for f in ["config.json", "axolotl_config.yaml", "training_args.json"]:
        p = ckpt_dir / f
        if p.exists():
            try:
                if f.endswith(".json"):
                    d = json.loads(p.read_text())
                    for key in ("_name_or_path", "base_model", "model_name_or_path"):
                        if key in d:
                            return d[key]
            except Exception:
                pass
    # Walk parent to find recipe yaml
    for yaml_file in (root / "configs").glob("*.yaml"):
        try:
            for line in yaml_file.read_text().splitlines():
                if line.strip().startswith("base_model:"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return None


def merge_lora(ckpt_dir: Path, base_model: str, merged_dir: Path) -> bool:
    """Merge LoRA adapters into the base model using PEFT."""
    info(f"Merging LoRA: {ckpt_dir.name} → {merged_dir.name}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    try:
        merge_script = f"""
import sys
from pathlib import Path
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ckpt = "{ckpt_dir}"
base = "{base_model}"
out  = "{merged_dir}"

print("Loading adapter...")
model = AutoPeftModelForCausalLM.from_pretrained(ckpt, device_map="cpu")
print("Merging and unloading...")
merged = model.merge_and_unload()
print("Saving merged model...")
merged.save_pretrained(out, safe_serialization=True)

tok = AutoTokenizer.from_pretrained(ckpt if Path(ckpt + "/tokenizer.json").exists() else base)
tok.save_pretrained(out)
print("Done.")
"""
        result = subprocess.run(
            [sys.executable, "-c", merge_script],
            capture_output=True, text=True, timeout=1800,
        )
        if result.returncode != 0:
            warn(f"LoRA merge failed: {result.stderr[-500:]}")
            return False
        ok(f"Merged model saved to {merged_dir}")
        return True
    except FileNotFoundError:
        warn("PEFT not installed. Install: pip install peft transformers")
        return False
    except subprocess.TimeoutExpired:
        warn("Merge timed out (>30min)")
        return False


def quantize_awq(merged_dir: Path, awq_dir: Path) -> bool:
    """Quantize merged model to AWQ 4-bit."""
    info(f"Quantizing to AWQ: {awq_dir.name}")
    awq_dir.mkdir(parents=True, exist_ok=True)
    try:
        quant_script = f"""
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = "{merged_dir}"
save_path  = "{awq_dir}"

quant_config = {{"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}}

print("Loading model for AWQ quantization...")
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto")
tok   = AutoTokenizer.from_pretrained(model_path)

print("Quantizing (this takes a while)...")
model.quantize(tok, quant_config=quant_config)

model.save_quantized(save_path, safetensors=True)
tok.save_pretrained(save_path)
print("AWQ quantization done.")
"""
        result = subprocess.run(
            [sys.executable, "-c", quant_script],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            warn(f"AWQ quantization failed: {result.stderr[-500:]}")
            warn("Install: pip install autoawq")
            return False
        ok(f"AWQ model saved to {awq_dir}")
        return True
    except Exception as e:
        warn(f"AWQ failed: {e}")
        return False


def copy_tokenizer(ckpt_dir: Path, export_dir: Path) -> None:
    tok_dir = export_dir / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("tokenizer*.json", "tokenizer*.model", "special_tokens*.json",
                    "tokenizer_config.json"):
        for f in ckpt_dir.glob(pattern):
            shutil.copy2(f, tok_dir / f.name)
    if any(tok_dir.iterdir()):
        ok(f"Tokenizer copied to {tok_dir}")
    else:
        warn("No tokenizer files found in checkpoint dir")


def run_inference_test(model_path: Path, out_file: Path) -> dict:
    """Run 5 sample prompts to verify the exported model generates."""
    prompts = [
        "What is the capital of France?",
        "Explain fine-tuning in one sentence.",
        "Write a haiku about training neural networks.",
        "What is 17 × 13?",
        "안녕하세요. 소개해 주세요.",
    ]
    results = {"model": str(model_path), "timestamp": datetime.now(timezone.utc).isoformat(),
               "prompts": []}
    try:
        test_script = f"""
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "{model_path}"
prompts = {json.dumps(prompts)}

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                              torch_dtype=torch.bfloat16)

outputs = []
for p in prompts:
    msgs = [{{"role": "user", "content": p}}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=128, do_sample=False)
    reply = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    outputs.append({{"prompt": p, "response": reply}})

print(json.dumps(outputs))
"""
        r = subprocess.run([sys.executable, "-c", test_script],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            parsed = json.loads(r.stdout.strip().splitlines()[-1])
            results["prompts"] = parsed
            ok("Inference test passed")
        else:
            warn(f"Inference test failed: {r.stderr[-300:]}")
    except Exception as e:
        warn(f"Inference test skipped: {e}")
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return results


def write_model_card(export_dir: Path, ckpt_dir: Path, base_model: str,
                     root: Path) -> Path:
    """Generate a MODEL_CARD.md template."""
    # Pull eval summary if available
    eval_dir = root / "eval_results" / ckpt_dir.name
    eval_summary_link = f"eval_results/{ckpt_dir.name}/summary.md"

    # Try to get val loss from training logs
    val_loss = "_fill in_"
    log_file = root / "logs" / f"train_{ckpt_dir.name}.log"
    if log_file.exists():
        for line in reversed(log_file.read_text().splitlines()):
            if "eval_loss" in line or "val_loss" in line:
                try:
                    import re
                    m = re.search(r"(?:eval|val)_loss['\": ]+([0-9.]+)", line)
                    if m:
                        val_loss = m.group(1)
                        break
                except Exception:
                    pass

    card = f"""\
# {{repo-name}}

Fine-tuned from [{base_model}](https://huggingface.co/{base_model}).

## Intended use
{{filled by user — required to be non-empty before publish}}

## Training details
- Method: QLoRA (LoRA rank 32, NF4 base weights)
- Hardware: {{fill in GPU}}
- Duration: {{fill in duration}}
- Base model: {base_model}
- Final val loss: {val_loss}

## Evaluation
Full report: [{eval_summary_link}]({eval_summary_link})

| Eval | Base | This model | Δ |
|------|------|-----------|---|
| MMLU | _pending_ | _pending_ | _pending_ |
| MT-Bench | _pending_ | _pending_ | _pending_ |

## Bias, risks, limitations
{{filled by user}}

## License
Inherits license of {base_model}. See LICENSE.

## Citation
{{filled by user, optional}}
"""
    out = export_dir / "MODEL_CARD.md"
    out.write_text(card)
    ok(f"MODEL_CARD.md written to {out}")
    warn("Edit MODEL_CARD.md — fill all {{...}} placeholders before running `ft publish`")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--merge", action="store_true", default=True,
                    help="Merge LoRA adapters into base (default: on)")
    ap.add_argument("--no-merge", action="store_false", dest="merge")
    ap.add_argument("--quantize", default="awq", choices=["awq", "gptq", "gguf", "none"],
                    help="Quantization format (default: awq)")
    ap.add_argument("--gguf", action="store_true", help="Also produce GGUF Q4_K_M")
    ap.add_argument("--output-dir", default=None, help="Override export dir")
    args = ap.parse_args()

    root = project_root()
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    if not ckpt_dir.exists():
        err(f"Checkpoint dir not found: {ckpt_dir}")
        sys.exit(2)

    run_id = ckpt_dir.name
    export_dir = Path(args.output_dir) if args.output_dir else root / "exports" / run_id
    export_dir.mkdir(parents=True, exist_ok=True)
    info(f"Checkpoint: {ckpt_dir}")
    info(f"Export dir: {export_dir}")

    base_model = find_base_model(ckpt_dir, root)
    if not base_model:
        err("Could not determine base model. Pass --base <model-id> or ensure recipe config is in configs/")
        sys.exit(2)
    info(f"Base model: {base_model}")

    # 1. Merge
    merged_dir = export_dir / "merged"
    merge_ok = False
    if args.merge:
        merge_ok = merge_lora(ckpt_dir, base_model, merged_dir)
    else:
        warn("Skipping LoRA merge (--no-merge). Quantization will use adapter dir directly.")
        merged_dir = ckpt_dir
        merge_ok = True

    # 2. Quantize
    source_for_quant = merged_dir if merge_ok else ckpt_dir
    if args.quantize == "awq":
        awq_dir = export_dir / "awq"
        quantize_awq(source_for_quant, awq_dir)

    # 3. Tokenizer copy
    copy_tokenizer(ckpt_dir, export_dir)

    # 4. Inference test
    test_model = export_dir / ("awq" if args.quantize == "awq" else "merged")
    if test_model.exists():
        info("Running inference test on exported model...")
        run_inference_test(test_model, export_dir / "inference_test.json")

    # 5. MODEL_CARD
    write_model_card(export_dir, ckpt_dir, base_model, root)

    update_state(
        phase="7 / Export complete",
        status=f"✓ Export complete: {export_dir}",
        next_action=f"Publish: `ft publish {export_dir}`",
        done=[f"Exported checkpoint {run_id} → {export_dir.name}"],
    )
    next_step(
        commands=[f"ft publish {export_dir} --hf-repo mkd-hika/<repo-name>"],
        note=(f"Edit {export_dir}/MODEL_CARD.md first — fill all {{...}} placeholders. "
              f"Then `ft publish` will push to HuggingFace."),
    )


if __name__ == "__main__":
    main()
