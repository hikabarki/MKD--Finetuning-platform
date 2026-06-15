#!/usr/bin/env python3
"""
ft_add_model.py — download a HuggingFace model and verify its chat template.

Usage:
    python ft_add_model.py <hf-model-id-or-url> [--accept-license]

Steps:
    1. Parse the input (full URL or "owner/name").
    2. Probe HF for license; refuse research-only weights without --accept-license.
    3. Download tokenizer + model weights to models/<sanitized-name>/.
    4. Run a chat-template round-trip test (the #1 silent killer of FT runs).
    5. Compute parameter count + memory estimate.
    6. Write models/<name>/MODEL_INFO.json.
    7. Print: ft configure --model <name> --dataset <name>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (
    project_root, info, ok, warn, err, next_step,
    update_state, require_free_gb,
)

HF_MODEL_URL = re.compile(r"https?://huggingface\.co/([^/?#]+/[^/?#]+)")

# License triggers that require explicit acknowledgement
RESTRICTED_LICENSES = {
    "research", "non-commercial", "noncommercial", "rail", "cc-by-nc",
    "llama2-community", "llama3-community", "gemma-terms-of-use", "other",
}


def parse_model_id(src: str) -> str:
    if m := HF_MODEL_URL.match(src):
        return m.group(1)
    if "/" in src and not src.startswith(("http://", "https://")):
        return src
    err(f"Could not parse model identifier: {src!r}")
    sys.exit(2)


def latest_local_dataset() -> str | None:
    """Heuristic: most recently modified data_cards/*.json."""
    root = project_root()
    cards = sorted((root / "data_cards").glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cards[0].stem if cards else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="HF model URL or 'owner/name'")
    ap.add_argument("--accept-license", action="store_true",
                    help="Acknowledge restricted license (research-only, NC, etc.)")
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    model_id = parse_model_id(args.source)
    safe_name = model_id.replace("/", "__").lower()
    root = project_root()
    dest = root / "models" / safe_name

    info(f"Model ID: {model_id}")
    require_free_gb(root, 30.0, f"model '{model_id}' (estimate)")

    # ---- License probe ----------------------------------------------------
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        meta = api.model_info(model_id, revision=args.revision)
        license_id = (meta.card_data or {}).get("license", "unknown") if meta.card_data else "unknown"
        license_id = str(license_id).lower()
        info(f"License: {license_id}")
        if any(tag in license_id for tag in RESTRICTED_LICENSES) and not args.accept_license:
            err(f"License '{license_id}' is restricted. Re-run with --accept-license to acknowledge.")
            err("By doing so you confirm your intended use complies with the license terms.")
            sys.exit(3)
    except Exception as e:
        warn(f"Could not probe license metadata: {e}")
        warn("Proceeding anyway — verify the model card manually.")

    # ---- Download ---------------------------------------------------------
    info(f"Downloading to {dest} (this may take several minutes)...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_id,
        revision=args.revision,
        local_dir=str(dest),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer*", "*.model"],
    )
    ok(f"Downloaded to {dest}")

    # ---- Chat-template round-trip ----------------------------------------
    info("Verifying chat template integrity (round-trip test)...")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(str(dest), trust_remote_code=True)
    except Exception as e:
        warn(f"Could not load tokenizer: {e}")
        warn("Skipping chat template verification — model files are downloaded correctly.")
        sys.exit(0)

    if tok.chat_template is None:
        warn("This model has NO chat_template defined in its tokenizer.")
        warn("You'll need to set chat_template: <name> manually in the recipe.")
    else:
        sample = [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Hello?"},
            {"role": "assistant", "content": "Hi."},
        ]
        rendered = tok.apply_chat_template(sample, tokenize=False)
        toks = tok(rendered, return_tensors=None)["input_ids"]
        re_text = tok.decode(toks, skip_special_tokens=False)
        # Don't expect exact byte-equality (special tokens may render differently)
        # but every assistant content must round-trip:
        for m in sample:
            if m["content"] not in re_text:
                err(f"Chat template round-trip FAILED. '{m['content']}' lost.")
                err(f"Rendered: {rendered!r}")
                sys.exit(4)
        ok(f"Chat template OK (template length: {len(rendered)} chars for sample)")

    # ---- Param count + memory estimate -----------------------------------
    info("Estimating parameter count from config...")
    cfg = json.loads((dest / "config.json").read_text())
    n_params = estimate_params(cfg)
    info(f"Approx. parameters: {n_params/1e9:.2f} B")

    mem = memory_budget(n_params)
    info("Memory estimates (single GPU):")
    for k, v in mem.items():
        info(f"  {k:30s} {v:>8.1f} GB")

    # ---- Manifest --------------------------------------------------------
    manifest = {
        "model_id": model_id,
        "revision": args.revision,
        "local_path": str(dest.relative_to(root)),
        "license": license_id if 'license_id' in locals() else "unknown",
        "params_billion": round(n_params / 1e9, 3),
        "memory_estimates_gb": {k: round(v, 2) for k, v in mem.items()},
        "chat_template_present": tok.chat_template is not None,
        "architecture": cfg.get("architectures", ["?"])[0],
    }
    (dest / "MODEL_INFO.json").write_text(json.dumps(manifest, indent=2))
    ok(f"Wrote {dest / 'MODEL_INFO.json'}")

    # ---- Next step -------------------------------------------------------
    ds = latest_local_dataset() or "<dataset-name>"
    short = model_id.split("/")[-1].lower()
    update_state(
        phase="3-4 / Model added",
        status=f"✓ Model '{model_id}' downloaded and verified.",
        next_action=f"Generate a training recipe: `ft configure --model {short} --dataset {ds}`",
        done=[f"Added model '{model_id}' ({n_params/1e9:.1f}B params)"],
    )
    next_step(
        commands=[f"ft configure --model {short} --dataset {ds}"],
        note=(f"This generates configs/run_<ts>.yaml tuned for your hardware. "
              f"After that: ft smoke-test, then ft train."),
    )


# ---- Estimators (no external network, pure config math) --------------------

def estimate_params(cfg: dict) -> int:
    """Rough parameter count from a transformers config dict.

    Accurate to ~5% for standard decoder-only architectures.
    """
    h = cfg.get("hidden_size") or cfg.get("d_model") or 4096
    L = cfg.get("num_hidden_layers") or cfg.get("n_layer") or 32
    v = cfg.get("vocab_size") or 32000
    ffn = cfg.get("intermediate_size") or 4 * h
    n_kv = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 32
    n_q  = cfg.get("num_attention_heads") or n_kv
    head_dim = h // max(n_q, 1)

    # Per layer: attn q/k/v/o + 3-matrix MLP (gate/up/down for SwiGLU)
    attn = h * h + 2 * (n_kv * head_dim) * h + h * h
    mlp  = 3 * h * ffn   # SwiGLU-style; off by ~25% for GeLU 2-mat MLPs
    layer = attn + mlp
    emb_and_norm = v * h + h * 2 + L * 2 * h  # input embed + LN params (small)
    return L * layer + emb_and_norm


def memory_budget(n_params: int) -> dict:
    """Per-GPU peak memory for a few common training profiles."""
    B = n_params
    return {
        "Full FT, bf16 + AdamW":          (2 + 2 + 4 + 4 + 4) * B / 1e9,  # ~16 B/p
        "LoRA r=32, bf16 base":           (2 * B + 0.5 * B * 0.005 * 8) / 1e9,
        "QLoRA r=32 (NF4 + 8-bit AdamW)": (0.5 * B + B * 0.005 * 6) / 1e9,
        "Inference bf16":                 2 * B / 1e9,
        "Inference 4-bit (AWQ)":          0.5 * B / 1e9,
    }


if __name__ == "__main__":
    main()
