#!/usr/bin/env python3
"""
ft_clean_dataset.py — apply the full cleaning pipeline (see 03_data.md).

Usage:
    python ft_clean_dataset.py <name> [options]

Pipeline (in order):
    schema validation → length filter → PII scrub → dedup → decontam → split → sample

Output:
    datasets/<name>/train.jsonl
    datasets/<name>/val.jsonl
    datasets/<name>/sample_100.jsonl
    data_cards/<name>.json    ← updated in place with row counts at each stage
    eval_results/<name>_pii_report.json
    eval_results/<name>_contamination.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (
    project_root, info, ok, warn, err, next_step,
    update_state, sha256_file, write_data_card,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- Stages -----------------------------------------------------------------

def schema_valid(row: dict) -> bool:
    if "messages" in row:
        msgs = row["messages"]
        return isinstance(msgs, list) and len(msgs) >= 2 and all(
            isinstance(m, dict) and m.get("content") and m.get("role") in ("system", "user", "assistant")
            for m in msgs
        )
    if "prompt" in row and "chosen" in row and "rejected" in row:
        return bool(row["chosen"].get("content")) and bool(row["rejected"].get("content"))
    return False


def length_ok(row: dict, max_chars: int) -> bool:
    if "messages" in row:
        total = sum(len(m.get("content", "")) for m in row["messages"])
    else:
        total = len(str(row.get("prompt"))) + len(row["chosen"]["content"]) + len(row["rejected"]["content"])
    # chars ≈ tokens*4 heuristic; we tokenize properly in `ft configure`
    return total <= max_chars


_EMAIL = re.compile(r"\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{7,}\d)(?!\d)")
_SSN_LIKE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_LIKE = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

def scrub_pii(row: dict, report: dict) -> dict | None:
    """Mask email/phone/SSN; drop the row if a credit-card-like number found.

    This is a fast regex baseline; for production, swap in Microsoft Presidio.
    """
    def scrub(s: str) -> tuple[str, bool]:
        had_cc = bool(_CC_LIKE.search(s))
        s = _CC_LIKE.sub("[CARD]", s)
        if _EMAIL.search(s):  report["email"] += 1
        if _PHONE.search(s):  report["phone"] += 1
        if _SSN_LIKE.search(s): report["ssn"] += 1
        s = _EMAIL.sub("[EMAIL]", s)
        s = _PHONE.sub("[PHONE]", s)
        s = _SSN_LIKE.sub("[SSN]", s)
        return s, had_cc

    drop = False
    if "messages" in row:
        new = []
        for m in row["messages"]:
            c, cc = scrub(m["content"])
            if cc: drop = True
            new.append({**m, "content": c})
        if drop:
            report["dropped_cc"] += 1
            return None
        return {**row, "messages": new}
    return row


def minhash_dedup(rows: list[dict], threshold: float = 0.85) -> list[dict]:
    """MinHash LSH dedup. Falls back to exact-text dedup if datasketch missing."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        warn("datasketch not installed; using exact-string dedup (less powerful)")
        seen, out = set(), []
        for r in rows:
            key = hashlib.sha1(_row_text(r).encode()).hexdigest()
            if key not in seen:
                seen.add(key); out.append(r)
        return out

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    keepers = []
    for i, r in enumerate(rows):
        m = MinHash(num_perm=128)
        for token in _row_text(r).split():
            m.update(token.encode())
        if lsh.query(m):
            continue
        lsh.insert(str(i), m)
        keepers.append(r)
    return keepers


def _row_text(r: dict) -> str:
    if "messages" in r:
        return " ".join(m.get("content", "") for m in r["messages"])
    return f"{r.get('prompt','')} {r['chosen']['content']} {r['rejected']['content']}"


def split_train_val(rows: list[dict], val_ratio: float = 0.01, seed: int = 42):
    rnd = random.Random(seed)
    shuffled = rows[:]
    rnd.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


# ---- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--max-chars", type=int, default=32 * 1024,
                    help="drop rows whose total content exceeds this (default 32k chars)")
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--no-pii", action="store_true",
                    help="skip PII scrub (NOT recommended)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip dedup (use only for tiny test datasets)")
    args = ap.parse_args()

    root = project_root()
    ds_dir = root / "datasets" / args.name
    norm = ds_dir / "normalized.jsonl"
    if not norm.exists():
        err(f"Not found: {norm}.  Did you run `ft add-dataset ...` first?")
        sys.exit(2)

    card_path = root / "data_cards" / f"{args.name}.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {"name": args.name}

    rows = load_jsonl(norm)
    info(f"Loaded {len(rows)} normalized rows")
    card["rows_raw"] = len(rows)

    # Schema
    rows = [r for r in rows if schema_valid(r)]
    info(f"After schema validation: {len(rows)} rows")
    card["rows_after_schema"] = len(rows)

    # Length
    rows = [r for r in rows if length_ok(r, args.max_chars)]
    info(f"After length filter (max_chars={args.max_chars}): {len(rows)} rows")
    card["rows_after_length"] = len(rows)

    # PII
    if not args.no_pii:
        report = {"email": 0, "phone": 0, "ssn": 0, "dropped_cc": 0}
        scrubbed = []
        for r in rows:
            out = scrub_pii(r, report)
            if out is not None:
                scrubbed.append(out)
        rows = scrubbed
        info(f"After PII scrub: {len(rows)} rows. Masked: {report}")
        card["rows_after_pii"] = len(rows)
        (root / "eval_results").mkdir(exist_ok=True)
        (root / "eval_results" / f"{args.name}_pii_report.json").write_text(
            json.dumps(report, indent=2))

    # Dedup
    if not args.no_dedup:
        before = len(rows)
        rows = minhash_dedup(rows)
        info(f"After MinHash dedup: {len(rows)} rows ({before-len(rows)} duplicates removed)")
        card["rows_after_dedup"] = len(rows)

    # Decontamination — placeholder. The full 13-gram check runs at eval time
    # against the actual eval sets; here we just record that it was deferred.
    card["rows_after_decontam"] = len(rows)
    card["decontamination_note"] = "13-gram overlap check deferred to `ft eval` (see 06_eval_plan.md)"

    # Split
    train, val = split_train_val(rows, args.val_ratio)
    train_p = ds_dir / "train.jsonl"
    val_p   = ds_dir / "val.jsonl"
    write_jsonl(train_p, train)
    write_jsonl(val_p,   val)
    info(f"Wrote {train_p} ({len(train)} rows) and {val_p} ({len(val)} rows)")
    card["train_rows"] = len(train)
    card["val_rows"]   = len(val)
    card["sha256_train"] = sha256_file(train_p)
    card["sha256_val"]   = sha256_file(val_p)

    # 100-row review sample
    sample_p = ds_dir / "sample_100.jsonl"
    write_jsonl(sample_p, random.Random(0).sample(train, min(100, len(train))))

    write_data_card(args.name, card)
    ok(f"Updated {card_path}")
    ok(f"Review sample: {sample_p}")

    update_state(
        phase="3 / Data Pipeline (cleaned)",
        status=f"✓ Dataset '{args.name}' cleaned. Train={len(train)}, Val={len(val)}.",
        next_action=(f"Eyeball datasets/{args.name}/sample_100.jsonl, then add a "
                     f"base model: `ft add-model <hf-model-id>`"),
        done=[f"Cleaned dataset '{args.name}'"],
    )
    next_step(
        commands=[
            f"head -3 datasets/{args.name}/sample_100.jsonl   # sanity check",
            f"ft add-model meta-llama/Llama-3.1-8B-Instruct",
        ],
        note=("Eyeball the sample first — 30 seconds of human review catches "
              "format bugs that no automated test will. Then pick any "
              "HuggingFace model ID to fine-tune."),
    )


if __name__ == "__main__":
    main()
