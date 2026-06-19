#!/usr/bin/env python3
"""
ft_clean_dataset.py — apply the full cleaning pipeline (see 03_data.md).

Usage:
    python ft_clean_dataset.py <name> [options]

Pipeline (in order):
    schema validation → language ID → quality heuristics → length filter → PII scrub → dedup → decontam → split → sample

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
    rows, errors = [], 0
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors += 1
            print(f"WARNING: Skipping malformed JSON on line {i}: {e}", flush=True)
            if errors > 100:
                print("ERROR: More than 100 JSON parse errors — aborting.", flush=True)
                sys.exit(1)
    return rows


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


# ---- Language identification ------------------------------------------------

# Korean Hangul Unicode blocks (covers syllables, jamo, compatibility jamo)
_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏ꥠ-꥿ힰ-퟿]")
# CJK unified ideographs (Chinese/Japanese)
_CJK    = re.compile(r"[一-鿿㐀-䶿]")


def detect_lang(text: str) -> str:
    """Return ISO 639-1 language code. Fast path: Unicode heuristic first,
    then langdetect if installed, then 'unknown'."""
    if not text.strip():
        return "unknown"
    # Heuristic: count script chars
    total = max(len(text), 1)
    ko_ratio = len(_HANGUL.findall(text)) / total
    cjk_ratio = len(_CJK.findall(text)) / total
    if ko_ratio > 0.05:
        return "ko"
    if cjk_ratio > 0.1:
        return "zh"
    try:
        from langdetect import detect as _ld_detect, LangDetectException
        try:
            return _ld_detect(text[:500])
        except LangDetectException:
            return "unknown"
    except ImportError:
        pass
    # Fallback: assume English if mostly ASCII
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / total
    return "en" if ascii_ratio > 0.8 else "unknown"


def tag_language(rows: list[dict]) -> list[dict]:
    """Add __lang__ field to every row. Does not drop anything."""
    out = []
    for r in rows:
        text = _row_text(r)[:800]
        out.append({**r, "__lang__": detect_lang(text)})
    return out


def filter_language(rows: list[dict], keep_langs: list[str]) -> tuple[list[dict], dict]:
    """Keep only rows whose __lang__ is in keep_langs. Returns (kept, report)."""
    report: dict[str, int] = {}
    kept = []
    for r in rows:
        lang = r.get("__lang__", "unknown")
        report[lang] = report.get(lang, 0) + 1
        if lang in keep_langs or "unknown" in keep_langs:
            kept.append(r)
    return kept, report


# ---- Quality heuristics -----------------------------------------------------

def _ngram_rep_ratio(text: str, n: int = 5) -> float:
    """Fraction of n-grams that are duplicates. High = repetitive/degenerate text."""
    words = text.split()
    if len(words) < n + 1:
        return 0.0
    grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def _garbled_ratio(text: str) -> float:
    """Fraction of chars that look garbled: non-printable, replacement chars, etc."""
    if not text:
        return 0.0
    bad = sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r")
    bad += text.count("�")   # Unicode replacement char
    bad += text.count("\x00")
    return bad / len(text)


def quality_ok(row: dict, min_response_chars: int, max_rep_ratio: float,
               max_garbled_ratio: float) -> tuple[bool, str]:
    """Return (passes, reason_if_dropped)."""
    if "messages" not in row:
        return True, ""
    msgs = row["messages"]

    # 1. Min assistant response length
    for m in msgs:
        if m.get("role") == "assistant":
            resp = m.get("content", "")
            if len(resp.strip()) < min_response_chars:
                return False, f"short_response ({len(resp.strip())} chars)"

    # 2. Repetition ratio on full text
    full = _row_text(row)
    rep = _ngram_rep_ratio(full)
    if rep > max_rep_ratio:
        return False, f"repetitive (rep_ratio={rep:.2f})"

    # 3. Garbled / encoding junk
    garbled = _garbled_ratio(full)
    if garbled > max_garbled_ratio:
        return False, f"garbled (garbled_ratio={garbled:.3f})"

    return True, ""


_EMAIL    = re.compile(r"\b[\w._%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE    = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{7,}\d)(?!\d)")
_SSN_LIKE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_LIKE  = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def _make_presidio_scrubber(lang: str = "en"):
    """Return a Presidio-based scrub function, or None if Presidio is not installed.

    M-7: If presidio_analyzer + presidio_anonymizer are available use them for
    production-grade NER-based PII detection.  The regex fallback remains for
    environments without Presidio.

    For Korean (ko) Presidio lacks a native model — fall back to regex-only scrub
    so that Korean PII is still partially detected rather than silently skipped.
    """
    # Korean has no Presidio NLP model; use regex path which handles phone/email/ID patterns.
    if lang == "ko":
        warn(
            "Korean (ko) detected — Presidio has no Korean NLP model. "
            "Using regex-only PII scrub. For production, add a custom Korean recognizer."
        )
        return None

    # Normalise language code (e.g. "en-US" → "en")
    presidio_lang = lang.split("-")[0] if lang else "en"

    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        _analyzer   = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()

        def _scrub_presidio(text: str) -> tuple[str, bool]:
            results = _analyzer.analyze(text=text, language=presidio_lang)
            had_cc  = any(r.entity_type == "CREDIT_CARD" for r in results)
            anon    = _anonymizer.anonymize(text=text, analyzer_results=results)
            return anon.text, had_cc

        return _scrub_presidio
    except ImportError:
        warn(
            "presidio_analyzer / presidio_anonymizer not installed — "
            "using regex-only PII scrub (NOT production-grade). "
            "Install with: pip install presidio-analyzer presidio-anonymizer && "
            "python -m spacy download en_core_web_lg"
        )
        return None


_presidio_scrubber = None   # set in main() once --lang is known


def scrub_pii(row: dict, report: dict) -> dict | None:
    """Mask PII; drop row if credit-card-like number found.

    Uses Microsoft Presidio when available (NER-based); falls back to regex.
    """

    def scrub(s: str) -> tuple[str, bool]:
        if _presidio_scrubber is not None:
            cleaned, had_cc = _presidio_scrubber(s)
            # Tally any entity placeholders that Presidio introduced
            if "[EMAIL_ADDRESS]" in cleaned or "[EMAIL]" in cleaned:
                report["email"] += 1
            if "[PHONE_NUMBER]" in cleaned or "[PHONE]" in cleaned:
                report["phone"] += 1
            if "[US_SSN]" in cleaned or "[SSN]" in cleaned:
                report["ssn"] += 1
            return cleaned, had_cc

        # Regex fallback
        had_cc = bool(_CC_LIKE.search(s))
        s = _CC_LIKE.sub("[CARD]", s)
        if _EMAIL.search(s):    report["email"] += 1
        if _PHONE.search(s):    report["phone"] += 1
        if _SSN_LIKE.search(s): report["ssn"]   += 1
        s = _EMAIL.sub("[EMAIL]", s)
        s = _PHONE.sub("[PHONE]", s)
        s = _SSN_LIKE.sub("[SSN]", s)
        return s, had_cc

    drop = False
    if "messages" in row:
        new = []
        for m in row["messages"]:
            c, cc = scrub(m["content"])
            if cc:
                drop = True
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
    # Language identification
    ap.add_argument("--lang", default="",
                    help="Comma-separated ISO 639-1 codes to keep, e.g. 'ko,en'. "
                         "Empty = keep all (just tag). 'unknown' always kept when listed.")
    ap.add_argument("--no-lang", action="store_true",
                    help="skip language identification entirely")
    # Quality heuristics
    ap.add_argument("--min-response-chars", type=int, default=10,
                    help="drop rows where assistant reply is shorter than this (default 10)")
    ap.add_argument("--max-rep-ratio", type=float, default=0.4,
                    help="drop rows where 5-gram repetition ratio exceeds this (default 0.4)")
    ap.add_argument("--max-garbled-ratio", type=float, default=0.02,
                    help="drop rows with this fraction of non-printable/junk chars (default 0.02)")
    ap.add_argument("--no-quality", action="store_true",
                    help="skip quality heuristic filtering")
    args = ap.parse_args()

    # Initialise Presidio scrubber with the dataset's primary language.
    # Korean (ko) has no Presidio model — falls back to regex-only scrub.
    global _presidio_scrubber
    pii_lang = (args.lang.split(",")[0].strip() if args.lang else "en") or "en"
    _presidio_scrubber = _make_presidio_scrubber(pii_lang)

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

    # Language identification
    if not args.no_lang:
        rows = tag_language(rows)
        lang_counts: dict[str, int] = {}
        for r in rows:
            l = r.get("__lang__", "unknown")
            lang_counts[l] = lang_counts.get(l, 0) + 1
        info(f"Language distribution: {dict(sorted(lang_counts.items(), key=lambda x: -x[1]))}")
        card["lang_distribution"] = lang_counts
        if args.lang:
            keep = [l.strip() for l in args.lang.split(",") if l.strip()]
            rows, _ = filter_language(rows, keep)
            info(f"After language filter (keep={keep}): {len(rows)} rows")
            card["rows_after_lang_filter"] = len(rows)
        # Strip internal tag before further processing
        rows = [{k: v for k, v in r.items() if k != "__lang__"} for r in rows]

    # Quality heuristics
    if not args.no_quality:
        quality_report = {"short_response": 0, "repetitive": 0, "garbled": 0}
        kept = []
        for r in rows:
            ok_flag, reason = quality_ok(
                r,
                min_response_chars=args.min_response_chars,
                max_rep_ratio=args.max_rep_ratio,
                max_garbled_ratio=args.max_garbled_ratio,
            )
            if ok_flag:
                kept.append(r)
            else:
                bucket = reason.split(" ")[0]
                quality_report[bucket] = quality_report.get(bucket, 0) + 1
        rows = kept
        info(f"After quality heuristics: {len(rows)} rows. Dropped: {quality_report}")
        card["rows_after_quality"] = len(rows)
        card["quality_report"] = quality_report
        (root / "eval_results").mkdir(exist_ok=True)
        (root / "eval_results" / f"{args.name}_quality_report.json").write_text(
            json.dumps(quality_report, indent=2))

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

    if len(rows) == 0:
        # Build a diagnostic showing which filter reduced the count the most.
        counts = {k: v for k, v in card.items() if k.startswith("rows_after_") or k == "rows_raw"}
        ordered = sorted(counts.items(), key=lambda x: x[1] if isinstance(x[1], int) else 999999)
        worst = ordered[0][0] if ordered else "unknown filter"
        err(
            "No rows remaining after filtering — training would fail.\n"
            f"  Biggest drop: {worst} = {ordered[0][1] if ordered else '?'} rows.\n"
            "  Try relaxing --min-response-chars, --max-chars, or --lang."
        )
        sys.exit(1)

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
