#!/usr/bin/env python3
"""
ft_add_dataset.py — download a dataset from anywhere and normalize to ChatML.

Usage:
    python ft_add_dataset.py <url-or-path> [--name <local-name>]

Supports:
    - HuggingFace dataset IDs:    https://huggingface.co/datasets/tatsu-lab/alpaca
                                  tatsu-lab/alpaca
    - Direct file URLs:           https://example.com/data.jsonl
    - Local paths:                ./mydata.jsonl  /abs/path/to/file.parquet

Output:
    datasets/<name>/raw/              ← exact downloaded files
    datasets/<name>/normalized.jsonl  ← ChatML JSONL, one example per line
    data_cards/<name>.json            ← provenance card

Then prints the next command:
    ft clean-dataset <name>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

# scripts/ adds itself to sys.path so siblings resolve when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (
    project_root, info, ok, warn, err, next_step,
    update_state, sha256_file, write_data_card, require_free_gb,
)


# ---- URL classification -----------------------------------------------------

HF_DATASET_URL = re.compile(r"https?://huggingface\.co/datasets/([^/?#]+/[^/?#]+)")
HF_SHORT_ID    = re.compile(r"^([\w\-.]+)/([\w\-.]+)$")

def classify_source(src: str) -> tuple[str, str]:
    """Return (kind, normalized_identifier)."""
    if Path(src).exists():
        return "local", str(Path(src).resolve())
    m = HF_DATASET_URL.match(src)
    if m:
        return "hf_dataset", m.group(1)
    if HF_SHORT_ID.match(src) and not src.startswith(("http://", "https://")):
        return "hf_dataset", src
    parsed = urlparse(src)
    if parsed.scheme in ("http", "https"):
        return "http_file", src
    err(f"Could not classify source: {src!r}")
    sys.exit(2)


# ---- Format detection & conversion -----------------------------------------

def to_chatml_rows(rows: list[dict]) -> tuple[list[dict], str]:
    """Detect schema and convert to ChatML.

    Returns (chatml_rows, detected_format).
    """
    if not rows:
        err("Empty dataset; nothing to convert")
        sys.exit(2)
    sample = rows[0]
    keys = set(sample.keys())

    # 1. Already ChatML
    if "messages" in keys and isinstance(sample["messages"], list):
        return [{"messages": r["messages"]} for r in rows], "chatml"

    # 2. DPO shape
    if {"prompt", "chosen", "rejected"} <= keys:
        out = []
        for r in rows:
            prompt = r["prompt"] if isinstance(r["prompt"], list) else [
                {"role": "user", "content": str(r["prompt"])}]
            out.append({
                "prompt": prompt,
                "chosen":   {"role": "assistant", "content": _as_text(r["chosen"])},
                "rejected": {"role": "assistant", "content": _as_text(r["rejected"])},
            })
        return out, "dpo"

    # 3. Alpaca
    if {"instruction", "output"} <= keys:
        out = []
        for r in rows:
            msgs = []
            if r.get("system"):
                msgs.append({"role": "system", "content": r["system"]})
            user = r["instruction"]
            if r.get("input"):
                user = f"{user}\n\n{r['input']}"
            msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": r["output"]})
            out.append({"messages": msgs})
        return out, "alpaca"

    # 4. ShareGPT
    if "conversations" in keys and isinstance(sample["conversations"], list):
        role_map = {"human": "user", "user": "user",
                    "gpt": "assistant", "assistant": "assistant",
                    "system": "system"}
        out = []
        for r in rows:
            msgs = []
            for turn in r["conversations"]:
                msgs.append({
                    "role": role_map.get(turn.get("from", "user"), "user"),
                    "content": turn.get("value", ""),
                })
            out.append({"messages": msgs})
        return out, "sharegpt"

    # 5. OpenAI prompt/completion
    if {"prompt", "completion"} <= keys:
        return [{"messages": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ]} for r in rows], "openai_pc"

    err(f"Unrecognized schema. Columns: {sorted(keys)}")
    err("Supported: messages | conversations | instruction+output | prompt+completion | prompt+chosen+rejected")
    sys.exit(2)


def _as_text(x):
    if isinstance(x, dict) and "content" in x:
        return x["content"]
    return str(x)


# ---- Source downloaders -----------------------------------------------------

def download_hf(ds_id: str, dest: Path) -> list[dict]:
    info(f"Downloading HuggingFace dataset: {ds_id}")
    from datasets import load_dataset       # heavy import, do it lazily
    ds = load_dataset(ds_id)
    # Prefer 'train' if available, else first split
    split = "train" if "train" in ds else next(iter(ds))
    info(f"Using split: {split}  ({len(ds[split])} rows)")
    rows = [dict(r) for r in ds[split]]
    (dest / f"{ds_id.replace('/', '__')}_{split}.json").write_text(
        json.dumps(rows[:50], indent=2))   # small preview, full lives in HF cache
    return rows


def download_http(url: str, dest: Path) -> list[dict]:
    import requests
    info(f"Downloading: {url}")
    fname = url.rstrip("/").split("/")[-1] or "data.bin"
    out = dest / fname
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with out.open("wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)
    return load_local(out)


def load_local(path: Path) -> list[dict]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if suf == ".json":
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else [data]
    if suf in (".parquet", ".pq"):
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    if suf == ".csv":
        import pandas as pd
        return pd.read_csv(path).to_dict(orient="records")
    err(f"Unsupported file extension: {suf}")
    sys.exit(2)


# ---- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="HF dataset URL/ID, HTTP URL, or local path")
    ap.add_argument("--name", help="Local name (default: derived from source)")
    args = ap.parse_args()

    root = project_root()
    kind, ident = classify_source(args.source)
    name = args.name or _derive_name(kind, ident)
    info(f"Source kind: {kind}    name: {name}")

    out_dir = root / "datasets" / name
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    require_free_gb(root, 5.0, f"dataset '{name}'")

    if kind == "hf_dataset":
        rows = download_hf(ident, raw_dir)
    elif kind == "http_file":
        rows = download_http(ident, raw_dir)
    else:  # local
        rows = load_local(Path(ident))

    info(f"Loaded {len(rows)} raw rows")
    chatml, fmt = to_chatml_rows(rows)
    info(f"Detected format: {fmt}    → normalized to ChatML ({len(chatml)} rows)")

    norm_path = out_dir / "normalized.jsonl"
    with norm_path.open("w") as f:
        for r in chatml:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    card = {
        "name": name,
        "source": args.source,
        "source_kind": kind,
        "source_id": ident,
        "detected_format": fmt,
        "rows_raw": len(rows),
        "rows_normalized": len(chatml),
        "normalized_path": str(norm_path.relative_to(root)),
        "sha256_normalized": sha256_file(norm_path),
        "license": "UNKNOWN — fill in before cleaning",
    }
    card_path = write_data_card(name, card)
    ok(f"Wrote {norm_path}")
    ok(f"Wrote {card_path}")

    if card["license"].startswith("UNKNOWN"):
        warn(f"License unknown. Open {card_path} and set 'license' before cleaning.")

    update_state(
        phase="3 / Data Pipeline (raw added)",
        status=f"✓ Dataset '{name}' downloaded and normalized to ChatML.",
        next_action=f"Clean and validate the dataset: `ft clean-dataset {name}`",
        done=[f"Added dataset '{name}' ({fmt}, {len(chatml)} rows)"],
    )
    next_step(
        commands=[f"ft clean-dataset {name}"],
        note=(f"This will dedupe, decontaminate, filter PII, and emit a 100-row "
              f"sample for human review at datasets/{name}/sample_100.jsonl."),
    )


def _derive_name(kind: str, ident: str) -> str:
    if kind == "hf_dataset":
        return ident.split("/", 1)[1].lower()
    if kind == "http_file":
        return Path(urlparse(ident).path).stem.lower() or "remote-dataset"
    return Path(ident).stem.lower()


if __name__ == "__main__":
    main()
