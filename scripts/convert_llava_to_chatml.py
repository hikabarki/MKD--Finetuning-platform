#!/usr/bin/env python3
"""Convert LLaVA-Instruct-150K JSON to ChatML JSONL.

Usage:
    python convert_llava_to_chatml.py <src.json> <out_dir>

The src.json file uses LLaVA's native format:
    [{"id": "...", "image": "...", "conversations": [{"from": "human"|"gpt", "value": "..."}, ...]}]

Output: out_dir/train.jsonl — one ChatML row per line:
    {"messages": [{"role": "user", "content": "..."}, ...], "image": "..."}
"""

import json
import sys
from pathlib import Path

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def convert(src: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    print(f"Loading {src} ({src.stat().st_size / 1e6:.0f} MB)...")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Converting {len(data)} records...")
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for item in data:
            convs = item.get("conversations") or []
            if not convs:
                skipped += 1
                continue

            messages = []
            for turn in convs:
                role = ROLE_MAP.get(turn.get("from", ""), "")
                if not role:
                    continue
                messages.append({"role": role, "content": turn.get("value", "")})

            if not messages:
                skipped += 1
                continue

            row = {"messages": messages}
            if item.get("image"):
                row["image"] = item["image"]

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Written: {written:,}  Skipped: {skipped:,}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <src.json> <out_dir>")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
