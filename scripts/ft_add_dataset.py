#!/usr/bin/env python3
"""
ft_add_dataset.py — download a dataset from anywhere and normalize to ChatML.

Usage:
    python ft_add_dataset.py <url-or-path> [--name <local-name>]
                             [--instruction-col <col>] [--response-col <col>]
                             [--chunk-size <chars>]

Supports:
    - HuggingFace dataset IDs:    https://huggingface.co/datasets/tatsu-lab/alpaca
                                  tatsu-lab/alpaca
    - Direct file URLs:           https://example.com/data.jsonl
    - Local paths:                ./mydata.jsonl  /abs/path/to/file.parquet
                                  ./corpus.txt

Detected formats (in priority order):
    chatml        messages list
    dpo           prompt + chosen + rejected
    alpaca        instruction + output (+ optional input/system)
    sharegpt      conversations list
    openai_pc     prompt + completion
    qa            question/query + answer/response (many column-name variants)
    summarization document/article/text + summary/abstract
    function_call messages with tool_calls / function_call
    aihub         Korean AI Hub JSON structure (info + data)
    plain_text    raw .txt — split into overlapping chunks
    custom        user-specified --instruction-col / --response-col

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

def to_chatml_rows(rows: list[dict], instruction_col: str = "",
                   response_col: str = "",
                   base_dir: "Path | None" = None) -> tuple[list[dict], str]:
    """Detect schema and convert to ChatML.

    Returns (chatml_rows, detected_format).
    """
    if not rows:
        err("Empty dataset; nothing to convert")
        sys.exit(2)
    sample = rows[0]
    keys = set(sample.keys())

    def _resolve_img(img_path: str) -> str:
        """Resolve a relative image path to absolute using base_dir."""
        if not img_path:
            return img_path
        p = Path(img_path)
        if p.is_absolute():
            return img_path
        if base_dir is not None:
            candidate = base_dir / p
            if candidate.exists():
                return str(candidate.resolve())
        return img_path

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
                    "system": "system", "bot": "assistant"}
        out = []
        for r in rows:
            msgs = []
            for turn in r["conversations"]:
                role = role_map.get(turn.get("from", "").lower(), "user")
                content = turn.get("value", turn.get("content", ""))
                if content:
                    msgs.append({"role": role, "content": content})
            if len(msgs) >= 2:
                out.append({"messages": msgs})
        return out, "sharegpt"

    # 5. OpenAI prompt/completion
    if {"prompt", "completion"} <= keys:
        return [{"messages": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["completion"]},
        ]} for r in rows], "openai_pc"

    # 6. QA format — many column-name variants
    QA_PAIRS = [
        ("question", "answer"), ("question", "answers"),
        ("query", "response"), ("query", "answer"),
        ("input", "output"), ("input", "target"),
        ("q", "a"), ("问题", "答案"), ("질문", "답변"),
    ]
    for q_col, a_col in QA_PAIRS:
        if q_col in keys and a_col in keys:
            out = []
            for r in rows:
                ans = r[a_col]
                # Some QA datasets store answers as a list (SQuAD-style)
                if isinstance(ans, dict) and "text" in ans:
                    ans = ans["text"][0] if isinstance(ans["text"], list) else ans["text"]
                elif isinstance(ans, list):
                    ans = ans[0] if ans else ""
                ans = str(ans).strip()
                question = str(r[q_col]).strip()
                ctx = r.get("context", r.get("passage", r.get("paragraph", "")))
                if ctx:
                    question = f"{question}\n\nContext:\n{ctx}"
                if question and ans:
                    out.append({"messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": ans},
                    ]})
            if out:
                return out, f"qa ({q_col}/{a_col})"

    # 7. Summarization format
    DOC_COLS  = ["document", "article", "text", "content", "body", "passage",
                 "source", "본문", "기사"]
    SUM_COLS  = ["summary", "abstract", "highlights", "tldr", "title",
                 "요약", "제목"]
    doc_col  = next((c for c in DOC_COLS if c in keys), None)
    sum_col  = next((c for c in SUM_COLS if c in keys), None)
    if doc_col and sum_col:
        out = []
        for r in rows:
            doc = str(r[doc_col]).strip()
            summ = str(r[sum_col]).strip()
            if doc and summ:
                out.append({"messages": [
                    {"role": "user",
                     "content": f"Summarize the following:\n\n{doc}"},
                    {"role": "assistant", "content": summ},
                ]})
        if out:
            return out, f"summarization ({doc_col}/{sum_col})"

    # 8. Function calling — messages list with tool_calls or function_call
    if "messages" in keys:
        # Already handled above, but tool-call rows may have non-list content
        pass
    if any("tool_calls" in str(r) or "function_call" in str(r) for r in rows[:20]):
        out = []
        for r in rows:
            msgs = r.get("messages", [])
            if msgs and isinstance(msgs, list):
                cleaned = []
                for m in msgs:
                    role = m.get("role", "user")
                    content = m.get("content") or ""
                    tool_calls = m.get("tool_calls")
                    if tool_calls:
                        content = content or json.dumps(tool_calls, ensure_ascii=False)
                    if content:
                        cleaned.append({"role": role, "content": str(content)})
                if len(cleaned) >= 2:
                    out.append({"messages": cleaned})
        if out:
            return out, "function_call"

    # 9. AI Hub (Korean government) format
    # Structure: {"info": {...}, "data": [{"header": {...}, "utterances": [...]}]}
    # or list of {"utterances": [...]} or {"dialogues": [...]}
    if isinstance(rows[0], dict):
        # Unwrap top-level info/data wrapper
        if "data" in keys and isinstance(rows[0]["data"], list):
            rows = rows[0]["data"]
            keys = set(rows[0].keys()) if rows else keys
        if "dialogues" in keys or "utterances" in keys or "dialog" in keys:
            dial_key = next((k for k in ("dialogues", "utterances", "dialog") if k in keys), None)
            out = []
            for r in rows:
                turns = r.get(dial_key, [])
                if not isinstance(turns, list):
                    continue
                msgs = []
                for t in turns:
                    role_raw = (t.get("role") or t.get("speaker") or
                                t.get("participantID") or "user")
                    role = "assistant" if str(role_raw).lower() in (
                        "sys", "system", "agent", "bot", "assistant", "ai", "2") else "user"
                    content = (t.get("text") or t.get("utterance") or
                               t.get("content") or "")
                    if content:
                        msgs.append({"role": role, "content": str(content)})
                if len(msgs) >= 2:
                    out.append({"messages": msgs})
            if out:
                return out, "aihub"

    # 10. Custom column mapper — user specified --instruction-col / --response-col
    if instruction_col and response_col:
        if instruction_col in keys and response_col in keys:
            out = []
            for r in rows:
                inst = str(r[instruction_col]).strip()
                resp = str(r[response_col]).strip()
                if inst and resp:
                    out.append({"messages": [
                        {"role": "user", "content": inst},
                        {"role": "assistant", "content": resp},
                    ]})
            if out:
                return out, f"custom ({instruction_col}/{response_col})"
        else:
            available = sorted(keys)
            err(f"--instruction-col '{instruction_col}' or --response-col '{response_col}' "
                f"not found.\nAvailable columns: {available}")
            sys.exit(2)

    # 10.5 Audio transcript — rows with "transcript"/"text" alongside audio metadata
    AUDIO_META_COLS = ["audio", "audio_path", "file", "filename", "path", "wav", "mp3"]
    AUDIO_TEXT_COLS = ["transcript", "transcription", "text", "content"]
    audio_col = next((c for c in AUDIO_META_COLS if c in keys), None)
    text_col   = next((c for c in AUDIO_TEXT_COLS if c in keys), None)
    if audio_col and text_col:
        out = []
        for r in rows:
            transcript = str(r.get(text_col, "")).strip()
            if transcript:
                out.append({"messages": [
                    {"role": "user",      "content": "Transcribe this audio."},
                    {"role": "assistant", "content": transcript},
                ]})
        if out:
            return out, f"audio_transcript ({text_col})"

    # 11. LLaVA format — {"id":..., "image":"path.jpg", "conversations":[...]}
    if "image" in keys and "conversations" in keys:
        role_map = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
        out = []
        for r in rows:
            msgs = []
            for turn in r.get("conversations", []):
                role = role_map.get(str(turn.get("from", "")).lower(), "user")
                content = str(turn.get("value", turn.get("content", ""))).replace("<image>", "").strip()
                if content:
                    msgs.append({"role": role, "content": content})
            if len(msgs) >= 2:
                row_out = {"messages": msgs}
                if r.get("image"):
                    row_out["images"] = [_resolve_img(str(r["image"]))]
                out.append(row_out)
        if out:
            return out, "llava"

    # 12. OpenAI vision format — content is a list with {"type":"image_url",...}
    if "messages" in keys:
        sample_msgs = sample.get("messages", [])
        has_vision = any(
            isinstance(m.get("content"), list) and
            any(c.get("type") in ("image_url", "image") for c in m["content"] if isinstance(c, dict))
            for m in sample_msgs
        )
        if has_vision:
            out = []
            for r in rows:
                msgs, images = [], []
                for m in r.get("messages", []):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    if isinstance(content, list):
                        texts, img_urls = [], []
                        for c in content:
                            if not isinstance(c, dict):
                                texts.append(str(c))
                                continue
                            if c.get("type") == "text":
                                texts.append(c.get("text", ""))
                            elif c.get("type") == "image_url":
                                url = c.get("image_url", {}).get("url", "")
                                if url:
                                    img_urls.append(url)
                            elif c.get("type") == "image":
                                img_urls.append(c.get("image", c.get("url", "")))
                        content = " ".join(texts).strip()
                        images.extend(img_urls)
                    if content:
                        msgs.append({"role": role, "content": content})
                if len(msgs) >= 2:
                    row_out = {"messages": msgs}
                    if images:
                        row_out["images"] = images
                    out.append(row_out)
            if out:
                return out, "openai_vision"

    # 13. Image-caption pairs — {"image": "path.jpg", "caption": "..."}
    CAPTION_COLS = ["caption", "text", "description", "label", "title", "요약", "설명"]
    cap_col = next((c for c in CAPTION_COLS if c in keys and c != "image"), None)
    if "image" in keys and cap_col:
        out = []
        for r in rows:
            img     = _resolve_img(str(r.get("image", "")))
            caption = str(r.get(cap_col, "")).strip()
            if caption:
                row_out = {"messages": [
                    {"role": "user",      "content": "Describe this image."},
                    {"role": "assistant", "content": caption},
                ]}
                if img:
                    row_out["images"] = [img]
                out.append(row_out)
        if out:
            return out, f"image_caption ({cap_col})"

    # 14. VQA format — {"image": "path.jpg", "question": "...", "answer": "..."}
    VQA_Q = ["question", "query", "질문"]
    VQA_A = ["answer", "answers", "multiple_choice_answer", "correct_answer",
             "response", "답변", "정답"]
    vqa_q = next((c for c in VQA_Q if c in keys), None)
    vqa_a = next((c for c in VQA_A if c in keys), None)
    if "image" in keys and vqa_q and vqa_a:
        out = []
        for r in rows:
            q   = str(r.get(vqa_q, "")).strip()
            ans = r.get(vqa_a, "")
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            ans = str(ans).strip()
            if q and ans:
                row_out = {"messages": [
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": ans},
                ]}
                if r.get("image"):
                    row_out["images"] = [_resolve_img(str(r["image"]))]
                out.append(row_out)
        if out:
            return out, f"vqa ({vqa_q}/{vqa_a})"

    # Nothing matched — show helpful error with available columns
    err(f"Unrecognized schema. Columns found: {sorted(keys)}")
    err("Supported formats: messages | conversations | instruction+output | "
        "prompt+completion | prompt+chosen+rejected | question+answer | "
        "document+summary | aihub dialogues | function_call | "
        "llava | openai_vision | image_caption | vqa")
    err("Tip: use --instruction-col <col> --response-col <col> to map custom columns.")
    sys.exit(2)


def _as_text(x):
    if isinstance(x, dict) and "content" in x:
        return x["content"]
    return str(x)


# ---- Source downloaders -----------------------------------------------------

def download_hf(ds_id: str, dest: Path) -> list[dict]:
    info(f"Downloading HuggingFace dataset: {ds_id}")
    from datasets import load_dataset
    ds = load_dataset(ds_id)
    split = "train" if "train" in ds else next(iter(ds))
    info(f"Using split: {split}  ({len(ds[split])} rows)")

    # Save PIL Images to disk so image-path columns work in vl_trainer
    img_dir = dest / "images"
    raw_rows = list(ds[split])
    rows: list[dict] = []
    pil_cols: list[str] = []

    for i, r in enumerate(raw_rows):
        d = dict(r)
        for col, val in list(d.items()):
            # Detect PIL Image objects (covers datasets.Image features)
            if hasattr(val, "save") and hasattr(val, "mode"):
                if col not in pil_cols:
                    pil_cols.append(col)
                    img_dir.mkdir(parents=True, exist_ok=True)
                    info(f"Detected image column '{col}' — saving images to {img_dir}")
                fname = img_dir / f"{i:06d}.jpg"
                try:
                    val.convert("RGB").save(str(fname), "JPEG", quality=90)
                    d[col] = str(fname)
                except Exception as _ie:
                    warn(f"Could not save image {i}: {_ie}")
                    d[col] = ""
        rows.append(d)

    if pil_cols:
        info(f"Saved {len(rows)} images for columns: {pil_cols}")

    (dest / f"{ds_id.replace('/', '__')}_{split}.json").write_text(
        json.dumps(rows[:50], indent=2))
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


def load_local(path: Path, chunk_size: int = 2048) -> list[dict]:
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if suf == ".json":
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return data if isinstance(data, list) else [data]
    if suf in (".parquet", ".pq"):
        import pandas as pd
        return pd.read_parquet(path).to_dict(orient="records")
    if suf in (".csv", ".tsv"):
        import pandas as pd
        sep = "\t" if suf == ".tsv" else ","
        return pd.read_csv(path, sep=sep).fillna("").to_dict(orient="records")
    if suf in (".xlsx", ".xls"):
        try:
            import pandas as pd
            df = pd.read_excel(path, engine="openpyxl" if suf == ".xlsx" else "xlrd")
            return df.fillna("").to_dict(orient="records")
        except ImportError:
            err("openpyxl not installed. Install with: pip install openpyxl")
            sys.exit(2)
    if suf == ".pdf":
        return _load_pdf(path, chunk_size)
    if suf == ".txt":
        return _chunk_text(path.read_text(encoding="utf-8", errors="replace"), chunk_size)
    err(f"Unsupported file extension: {suf}. Supported: .jsonl .json .parquet .csv .tsv .txt .pdf .xlsx .xls")
    sys.exit(2)


def _load_pdf(path: Path, chunk_size: int = 2048) -> list[dict]:
    """Extract text+images from a PDF.

    With pymupdf: renders each page as a JPEG and pairs it with that page's
    text → image-caption rows usable by vl_trainer.
    Falls back to text-only chunking if pymupdf is unavailable.
    """
    try:
        import fitz  # pymupdf
        img_dir = path.parent / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(str(path))
        rows = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            # Render page to image
            pix = page.get_pixmap(dpi=150)
            img_path = img_dir / f"{path.stem}_page{i+1:04d}.jpg"
            pix.save(str(img_path))
            if text:
                rows.append({
                    "messages": [
                        {"role": "user",      "content": "Describe what you see on this page."},
                        {"role": "assistant", "content": text},
                    ],
                    "images": [str(img_path)],
                })
            else:
                # Image-only page (scanned) — ask model to describe it
                rows.append({
                    "messages": [
                        {"role": "user",      "content": "Describe what you see on this page."},
                        {"role": "assistant", "content": ""},
                    ],
                    "images": [str(img_path)],
                })
        doc.close()
        rows = [r for r in rows if r["messages"][1]["content"]]  # drop blank-text pages
        if rows:
            info(f"PDF → {len(rows)} image+text pages from {path.name}")
            return rows
    except ImportError:
        pass

    # Fallback: text-only extraction
    text = ""
    try:
        from pdfminer.high_level import extract_text as _pdfminer_extract
        text = _pdfminer_extract(str(path))
    except ImportError:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            err("No PDF library found. Install: pip install pymupdf")
            sys.exit(2)
    if not text.strip():
        err(f"PDF appears empty or unreadable: {path}")
        sys.exit(2)
    info(f"Extracted {len(text)} chars (text only) from PDF: {path.name}")
    return _chunk_text(text, chunk_size)


def _chunk_text(text: str, chunk_size: int = 2048) -> list[dict]:
    """Split plain text into overlapping chunks wrapped as ChatML rows.

    Strategy:
      - Split on paragraph breaks first (double newline)
      - Accumulate paragraphs into chunks up to chunk_size chars
      - Overlap: last paragraph of previous chunk becomes first of next
    Each chunk becomes: user asks "Continue the text:" → assistant returns chunk.
    Adjacent chunks become instruction→response pairs so the model learns continuation.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Overlap: keep last paragraph
            current = [current[-1], para]
            current_len = len(current[-2]) + len(para)
        else:
            current.append(para)
            current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))

    if len(chunks) < 2:
        # Single chunk — wrap as standalone completion
        return [{"messages": [
            {"role": "user", "content": "Please continue writing based on the following text:"},
            {"role": "assistant", "content": chunks[0]},
        ]}]

    # Pair consecutive chunks: chunk[i] as context → chunk[i+1] as continuation
    rows = []
    for i in range(len(chunks) - 1):
        rows.append({"messages": [
            {"role": "user",
             "content": f"Continue the following text:\n\n{chunks[i]}"},
            {"role": "assistant", "content": chunks[i + 1]},
        ]})
    return rows


# ---- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="HF dataset URL/ID, HTTP URL, or local path")
    ap.add_argument("--name", help="Local name (default: derived from source)")
    ap.add_argument("--instruction-col", default="",
                    help="Column to use as the instruction/question (for custom schemas)")
    ap.add_argument("--response-col", default="",
                    help="Column to use as the response/answer (for custom schemas)")
    ap.add_argument("--chunk-size", type=int, default=2048,
                    help="Max chars per chunk for plain-text files (default: 2048)")
    args = ap.parse_args()

    root = project_root()
    kind, ident = classify_source(args.source)
    name = args.name or _derive_name(kind, ident)
    if not name or "/" in name or ".." in name or "\x00" in name:
        err(f"Invalid dataset name: {name!r}"); sys.exit(1)
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
        rows = load_local(Path(ident), chunk_size=args.chunk_size)

    info(f"Loaded {len(rows)} raw rows")
    # Plain text files come back pre-chunked as ChatML rows
    src_ext = Path(ident).suffix.lower() if kind == "local" else Path(args.source.split("?")[0]).suffix.lower()
    if src_ext == ".txt" and rows and isinstance(rows[0], dict) and "messages" in rows[0]:
        chatml, fmt = rows, "plain_text"
    else:
        # Pass base_dir so relative image paths in local datasets resolve correctly
        _base_dir = Path(ident).parent if kind == "local" else None
        chatml, fmt = to_chatml_rows(
            rows,
            instruction_col=args.instruction_col,
            response_col=args.response_col,
            base_dir=_base_dir,
        )
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
