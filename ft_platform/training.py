"""
platform/training.py — Shared VL training subprocess helper.

Extracted from run_user_quickstart and run_user_custommix in app.py.
Phase 3: stub — full extraction in next iteration.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Iterator


def _run_vl_training_subprocess(
    cmd: list[str],
    run_id: str,
    output_fn: Callable[[str], str],
) -> Iterator[str]:
    """
    Run a VL training subprocess, yielding output_fn(stage) for each stage.

    cmd      — full command list (e.g. ["python", "vl_trainer.py", ...])
    run_id   — short identifier used for log file naming
    output_fn — callable(stage_name) -> HTML snippet to yield at each stage transition
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is None:
        return
    for line in proc.stdout:
        line = line.rstrip()
        # Yield progress updates based on key lines
        if "Epoch" in line or "step" in line.lower():
            yield output_fn("train")
        elif "eval" in line.lower() or "accuracy" in line.lower():
            yield output_fn("eval")
        elif "export" in line.lower() or "saving" in line.lower():
            yield output_fn("export")
    proc.wait()
