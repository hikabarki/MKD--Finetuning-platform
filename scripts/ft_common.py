"""
ft_common.py — shared utilities for the ft-* CLI scripts.

Provides the "what to run next" printer that is the core UX of the platform.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Iterable

# ---- Project root resolution ------------------------------------------------

def project_root() -> Path:
    """The project folder root. Resolved from $FT_PROJECT or the cwd."""
    env = os.environ.get("FT_PROJECT")
    if env:
        return Path(env).resolve()
    # walk up looking for STATE.md
    cur = Path.cwd().resolve()
    for p in [cur, *cur.parents]:
        if (p / "STATE.md").exists():
            return p
    return cur  # fall back; let the caller create files

# ---- Pretty printing --------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_C = {
    "reset":  "\033[0m" if _USE_COLOR else "",
    "bold":   "\033[1m" if _USE_COLOR else "",
    "dim":    "\033[2m" if _USE_COLOR else "",
    "green":  "\033[32m" if _USE_COLOR else "",
    "yellow": "\033[33m" if _USE_COLOR else "",
    "red":    "\033[31m" if _USE_COLOR else "",
    "cyan":   "\033[36m" if _USE_COLOR else "",
    "mag":    "\033[35m" if _USE_COLOR else "",
}

def info(msg: str) -> None: print(f"{_C['cyan']}ℹ {msg}{_C['reset']}")
def ok(msg: str)   -> None: print(f"{_C['green']}✓ {msg}{_C['reset']}")
def warn(msg: str) -> None: print(f"{_C['yellow']}⚠ {msg}{_C['reset']}")
def err(msg: str)  -> None: print(f"{_C['red']}✗ {msg}{_C['reset']}", file=sys.stderr)

def next_step(commands: list[str], note: str = "") -> None:
    """Print the canonical 'what to run next' block.

    In UI mode (FT_UI=1) this is suppressed — the tabs guide the user instead.
    In CLI mode this prints the colored box.
    """
    if os.environ.get("FT_UI") == "1":
        return

    width = max(72, *(len(c) for c in commands)) + 6
    bar = "─" * width
    print()
    print(f"{_C['bold']}{_C['green']}╭{bar}╮{_C['reset']}")
    title = "NEXT STEP — copy and run this:"
    pad = " " * (width - len(title) - 1)
    print(f"{_C['bold']}{_C['green']}│ {title}{pad}│{_C['reset']}")
    print(f"{_C['bold']}{_C['green']}├{bar}┤{_C['reset']}")
    for cmd in commands:
        pad = " " * (width - len(cmd) - 4)
        print(f"{_C['bold']}{_C['green']}│{_C['reset']}  {_C['cyan']}{cmd}{_C['reset']}{pad}{_C['bold']}{_C['green']}│{_C['reset']}")
    if note:
        print(f"{_C['bold']}{_C['green']}├{bar}┤{_C['reset']}")
        for line in textwrap.wrap(note, width - 4):
            pad = " " * (width - len(line) - 4)
            print(f"{_C['bold']}{_C['green']}│{_C['reset']}  {_C['dim']}{line}{_C['reset']}{pad}{_C['bold']}{_C['green']}│{_C['reset']}")
    print(f"{_C['bold']}{_C['green']}╰{bar}╯{_C['reset']}")
    print()

# ---- STATE.md updater (the protocol contract) ------------------------------

def update_state(phase: str, status: str, next_action: str, done: list[str] | None = None) -> None:
    """Append-aware STATE.md updater.

    Rewrites STATE.md in place but DOES NOT touch DECISIONS.md or eval_results/.
    """
    from datetime import datetime, timezone
    root = project_root()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    done_lines = "\n".join(f"- ✅ {d}" for d in (done or []))
    content = textwrap.dedent(f"""\
        # STATE

        **Updated:** {stamp} by ft-cli
        **Phase:** {phase}
        **Status:** {status}

        ## Recent actions
        {done_lines or '(none)'}

        ## Next action
        {next_action}
        """)
    (root / "STATE.md").write_text(content)

# ---- Hashes / manifests -----------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_data_card(name: str, card: dict) -> Path:
    root = project_root()
    out = root / "data_cards" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2))
    return out

# ---- Disk-space pre-flight --------------------------------------------------

def require_free_gb(path: Path, gb: float, what: str) -> None:
    total, used, free = shutil.disk_usage(path)
    free_gb = free / 1024**3
    if free_gb < gb:
        err(f"Not enough disk on {path}: {free_gb:.1f} GB free, need {gb:.1f} GB for {what}")
        sys.exit(1)
