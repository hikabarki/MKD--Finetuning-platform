#!/usr/bin/env python3
"""
ft_watch.py — live training monitor: loss, grad_norm, tok/s, GPU, ETA.

Usage:
    ft watch <run-name>
    ft watch run_2026-05-27_0039
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import project_root

_C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "green":  "\033[32m",
    "yellow": "\033[33m",
    "cyan":   "\033[36m",
    "red":    "\033[31m",
    "dim":    "\033[2m",
}


def gpu_stats() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        ).strip().splitlines()[0]
        util, mem_used, mem_total, temp, pwr = [p.strip() for p in out.split(",")]
        return (f"GPU {util}%  VRAM {mem_used}/{mem_total} MiB  "
                f"Temp {temp}°C  Power {pwr}W")
    except Exception:
        return "GPU stats unavailable"


def parse_step_line(line: str) -> dict | None:
    """Extract step metrics from Axolotl / HF Trainer log lines."""
    # HF Trainer JSON log: {'loss': 1.234, 'grad_norm': 0.85, 'learning_rate': 2e-4, 'epoch': 1.0, 'step': 100}
    m = re.search(r"\{['\"]loss['\"].*?\}", line)
    if not m:
        return None
    try:
        raw = m.group(0).replace("'", '"')
        import json
        d = json.loads(raw)
        if "loss" in d:
            return d
    except Exception:
        pass
    return None


def draw_bar(val: float, max_val: float, width: int = 20, color: str = "green") -> str:
    filled = int(min(val / max(max_val, 1e-9), 1.0) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{_C[color]}{bar}{_C['reset']}"


def main():
    if len(sys.argv) < 2:
        print("Usage: ft watch <run-name>")
        sys.exit(2)

    run_name = sys.argv[1]
    root = project_root()
    log_file = root / "logs" / f"train_{run_name}.log"

    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        print("Training may still be starting up — retrying every 3s...")

    print(f"\n{_C['bold']}{_C['cyan']}  ft watch — {run_name}{_C['reset']}")
    print(f"{_C['dim']}  Press Ctrl+C to exit{_C['reset']}\n")

    # State
    losses: deque = deque(maxlen=50)
    steps: list = []
    last_size = 0
    last_step_time = time.time()
    step_times: deque = deque(maxlen=10)
    last_step = 0

    try:
        while True:
            if not log_file.exists():
                time.sleep(3)
                continue

            current_size = log_file.stat().st_size
            if current_size == last_size:
                time.sleep(2)
                continue

            # Read new lines
            with log_file.open() as f:
                f.seek(last_size)
                new_lines = f.read()
            last_size = current_size

            for line in new_lines.splitlines():
                metrics = parse_step_line(line)
                if not metrics:
                    continue

                loss = float(metrics.get("loss", 0))
                grad = float(metrics.get("grad_norm", 0))
                lr = float(metrics.get("learning_rate", 0))
                epoch = float(metrics.get("epoch", 0))
                step = int(metrics.get("step", 0))

                if step > last_step:
                    now = time.time()
                    step_times.append(now - last_step_time)
                    last_step_time = now
                    last_step = step

                losses.append(loss)
                steps.append(step)

                # Compute tok/s estimate
                tok_s = "—"
                if step_times:
                    avg_step_t = sum(step_times) / len(step_times)
                    # rough: seq_len * micro_batch / step_time
                    tok_s = f"~{int(2048 / avg_step_t):,}"

                # Loss trend
                trend = ""
                if len(losses) >= 5:
                    recent = list(losses)[-5:]
                    if recent[-1] < recent[0]:
                        trend = f"{_C['green']}↓{_C['reset']}"
                    elif recent[-1] > recent[0] * 1.05:
                        trend = f"{_C['red']}↑{_C['reset']}"
                    else:
                        trend = f"{_C['yellow']}→{_C['reset']}"

                # Loss color
                loss_color = "green" if loss < 1.5 else ("yellow" if loss < 3.0 else "red")

                # Clear and redraw
                print("\033[2J\033[H", end="")  # clear screen
                print(f"{_C['bold']}{_C['cyan']}  ╔══ ft watch — {run_name} ══╗{_C['reset']}")
                print(f"  {_C['dim']}Updated: {time.strftime('%H:%M:%S')}{_C['reset']}\n")

                print(f"  Step    {_C['bold']}{step:>6}{_C['reset']}")
                print(f"  Epoch   {_C['bold']}{epoch:.2f}{_C['reset']}")
                print(f"  Loss    {_C[loss_color]}{_C['bold']}{loss:.4f}{_C['reset']}  {trend}  "
                      f"{draw_bar(max(0, 3.0 - loss), 3.0, 25, loss_color)}")
                print(f"  Grad    {_C['bold']}{grad:.3f}{_C['reset']}  "
                      f"{'⚠ HIGH' if grad > 5 else ''}")
                print(f"  LR      {_C['dim']}{lr:.2e}{_C['reset']}")
                print(f"  Tok/s   {_C['cyan']}{tok_s}{_C['reset']}")
                print()
                print(f"  GPU     {_C['dim']}{gpu_stats()}{_C['reset']}")

                # Mini loss sparkline
                if len(losses) >= 2:
                    mini = list(losses)[-30:]
                    mn, mx = min(mini), max(mini)
                    spark = ""
                    bars = " ▁▂▃▄▅▆▇█"
                    for v in mini:
                        idx = int((v - mn) / max(mx - mn, 1e-9) * 8)
                        spark += bars[min(idx, 8)]
                    print(f"\n  Loss history: {_C['dim']}{spark}{_C['reset']}")

                print(f"\n  {_C['dim']}Ctrl+C to exit{_C['reset']}")

            time.sleep(2)

    except KeyboardInterrupt:
        print(f"\n\n{_C['dim']}  Stopped watching. Training continues in background.{_C['reset']}\n")


if __name__ == "__main__":
    main()
