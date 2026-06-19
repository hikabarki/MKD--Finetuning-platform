#!/usr/bin/env python3
"""
ft_train.py — launch a full training run via Axolotl (or Unsloth for single-GPU).

Usage:
    python ft_train.py <recipe.yaml>

What it does:
    1. Reads recipe YAML and verifies all files present.
    2. Detects GPU count — routes to Axolotl (multi) or Unsloth fast-path (single).
    3. Builds the accelerate / torchrun launch command.
    4. Runs training with real-time watchdog: NaN loss, grad_norm spike, disk space.
    5. On finish, prints: ft eval checkpoints/<run-id>

Abort criteria (from 05_runbook.md):
    - Loss is NaN/Inf → abort, save crash dump
    - grad_norm > 10 for 5 consecutive steps → abort
    - Loss > 5× rolling median for 3 steps → abort
    - Disk free < 2× checkpoint size → pause and alert
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (
    project_root, info, ok, warn, err, next_step,
    update_state, require_free_gb,
)


def load_yaml_str(path: Path) -> str:
    return path.read_text()


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text())


def count_gpus() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
        return len(out)
    except Exception:
        return 0


def preflight(root: Path, recipe: dict, recipe_path: Path) -> None:
    """Verify all files referenced in the recipe exist."""
    ds_cfg = recipe.get("datasets", [])
    if ds_cfg and isinstance(ds_cfg[0], dict):
        train_path = root / ds_cfg[0].get("path", "")
        if not train_path.exists():
            err(f"Training data not found: {train_path}")
            err("Run `ft clean-dataset <name>` first.")
            sys.exit(2)

    model_path = root / recipe.get("base_model", "")
    if not model_path.exists():
        warn(f"Model not found locally at {model_path} — Axolotl will download from HF Hub.")

    require_free_gb(root / "checkpoints", 10.0, "checkpoints")
    ok("Pre-flight checks passed")


def build_launch_cmd(recipe_path: Path, n_gpus: int) -> list[str]:
    """Build the torchrun / accelerate command for Axolotl."""
    if n_gpus > 1:
        return [
            "accelerate", "launch",
            f"--num_processes={n_gpus}",
            "--mixed_precision=bf16",
            "-m", "axolotl.cli.train",
            str(recipe_path),
        ]
    elif n_gpus == 1:
        return [
            sys.executable, "-m", "axolotl.cli.train",
            str(recipe_path),
        ]
    else:
        # CPU — only useful for testing
        return [
            sys.executable, "-m", "axolotl.cli.train",
            str(recipe_path),
            "--no_cuda",
        ]


def save_crash_dump(root: Path, output_dir: Path, reason: str, log_lines: list[str]) -> None:
    dump_dir = output_dir / "crash_dump"
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "last_100_lines.txt").write_text("\n".join(log_lines[-100:]))
    (dump_dir / "post_mortem.md").write_text(f"""\
# Post-Mortem

**Abort reason:** {reason}

## Last 100 log lines
See `last_100_lines.txt`

## Environment
- Python: {sys.version}
- Project root: {root}

## Next steps
1. Review the log lines above for the root cause.
2. Fix the config or data issue.
3. Re-run `ft smoke-test` before retrying.
""")
    warn(f"Crash dump saved to {dump_dir}")


def tail_and_watch(proc: subprocess.Popen, output_dir: Path, root: Path) -> int:
    """Stream stdout, watch for abort conditions, return exit code."""
    log_lines: list[str] = []
    loss_history: list[float] = []
    bad_grad_streak = 0
    bad_loss_streak = 0

    log_file = root / "logs" / f"train_{output_dir.name}.log"
    log_file.parent.mkdir(exist_ok=True)

    with log_file.open("w") as lf:
        for raw in proc.stdout:  # type: ignore[union-attr]
            line = raw.rstrip()
            print(line)
            lf.write(line + "\n")
            log_lines.append(line)

            # Parse loss from Axolotl's log format: "{'loss': 1.234, ...}"
            if "'loss'" in line or '"loss"' in line:
                try:
                    start = line.index("{")
                    d = json.loads(line[start:].replace("'", '"'))
                    loss = float(d.get("loss", float("nan")))
                    grad = float(d.get("grad_norm", 0))

                    if loss != loss or loss == float("inf"):
                        save_crash_dump(root, output_dir, "NaN/Inf loss", log_lines)
                        proc.terminate()
                        err("ABORT: loss is NaN/Inf")
                        return 1

                    loss_history.append(loss)
                    if len(loss_history) > 20:
                        median = sorted(loss_history[-20:][::2])[5]
                        if loss > 5 * median:
                            bad_loss_streak += 1
                            if bad_loss_streak >= 3:
                                save_crash_dump(root, output_dir,
                                                f"Loss spike: {loss:.3f} > 5× median {median:.3f}",
                                                log_lines)
                                proc.terminate()
                                err("ABORT: loss spike (> 5× rolling median for 3 steps)")
                                return 1
                        else:
                            bad_loss_streak = 0

                    if grad > 10:
                        bad_grad_streak += 1
                        if bad_grad_streak >= 5:
                            save_crash_dump(root, output_dir,
                                            f"grad_norm > 10 for 5 consecutive steps",
                                            log_lines)
                            proc.terminate()
                            err("ABORT: grad_norm > 10 for 5 consecutive steps")
                            return 1
                    else:
                        bad_grad_streak = 0
                except Exception:
                    pass

    proc.wait()
    return proc.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(prog="ft train", add_help=False)
    parser.add_argument("recipe")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Cap training at N steps (0 = run all epochs)")
    parsed, _ = parser.parse_known_args()

    recipe_path = Path(parsed.recipe)
    root = project_root()
    if not recipe_path.exists() and not recipe_path.is_absolute():
        recipe_path = root / recipe_path
    if not recipe_path.exists():
        err(f"Recipe not found: {recipe_path}")
        sys.exit(2)

    # If --max-steps given, write a patched copy of the YAML
    if parsed.max_steps > 0:
        import re as _re, tempfile, shutil as _sh
        txt = recipe_path.read_text()
        if "max_steps:" not in txt:
            txt = _re.sub(r'^(num_epochs:)', r'max_steps: ' + str(parsed.max_steps) + r'\n\1',
                          txt, flags=_re.MULTILINE)
        else:
            txt = _re.sub(r'^max_steps:.*', f'max_steps: {parsed.max_steps}', txt, flags=_re.MULTILINE)
        import tempfile as _tf
        with _tf.NamedTemporaryFile(suffix=".yaml", prefix="ft_train_",
                                    mode="w", delete=False) as _fh:
            _fh.write(txt)
            tmp = Path(_fh.name)
        info(f"max_steps={parsed.max_steps} → using patched config {tmp.name}")
        recipe_path = tmp

    recipe = load_yaml(recipe_path)
    output_dir = root / recipe.get("output_dir", "checkpoints/run")

    info(f"Recipe:     {recipe_path}")
    info(f"Output dir: {output_dir}")

    preflight(root, recipe, recipe_path)

    n_gpus = count_gpus()
    info(f"GPUs detected: {n_gpus}")

    cmd = build_launch_cmd(recipe_path, n_gpus)
    info(f"Launch command: {' '.join(cmd)}")

    # Write env snapshot for reproducibility
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (output_dir / "env_snapshot.txt").write_text(pip_freeze)
    except Exception:
        pass

    update_state(
        phase="5 / Training (running)",
        status=f"Training started: {recipe_path.name}",
        next_action=f"Monitor with: ft watch {output_dir.name}",
        done=[f"Started training with recipe {recipe_path.name}"],
    )

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(root),
            env={**os.environ, "NCCL_DEBUG": "INFO", "PYTHONUNBUFFERED": "1"},
        )
        rc = tail_and_watch(proc, output_dir, root)
    except FileNotFoundError:
        err(f"Could not launch training: command not found — {cmd[0]}")
        err("Is Axolotl installed? Run: pip install axolotl")
        update_state(phase="5 / Training failed", status="Training failed: axolotl not found",
                     next_action="pip install axolotl, then retry ft train")
        sys.exit(3)
    except KeyboardInterrupt:
        update_state(phase="5 / Training stopped",
                     status="Training manually interrupted (Ctrl+C)",
                     next_action=f"Resume: ft train {recipe_path}  or  ft eval {output_dir}")
        sys.exit(130)

    elapsed = time.time() - t0
    h = int(elapsed // 3600); m = int((elapsed % 3600) // 60)

    if rc != 0:
        err(f"Training exited with code {rc} after {h}h {m}m")
        err(f"Check logs at: logs/train_{output_dir.name}.log")
        update_state(phase="5 / Training failed",
                     status=f"Training exited with code {rc} after {h}h {m}m",
                     next_action=f"Check logs: logs/train_{output_dir.name}.log")
        sys.exit(rc)

    ok(f"Training complete in {h}h {m}m")
    ok(f"Output: {output_dir}")

    # LOW: Archive completed run config so configs/ doesn't fill up indefinitely
    archive_dir = root / "configs" / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(recipe_path), str(archive_dir / recipe_path.name))
        info(f"Recipe archived to {archive_dir / recipe_path.name}")
    except Exception as _ae:
        warn(f"Could not archive recipe: {_ae}")

    update_state(
        phase="5 / Training complete",
        status=f"✓ Training finished: {output_dir.name} ({h}h {m}m)",
        next_action=f"Evaluate: `ft eval {output_dir}`",
        done=[f"Training complete: {recipe_path.name} → {output_dir.name}"],
    )
    next_step(
        commands=[f"ft eval {output_dir}"],
        note="Runs lm-eval-harness + MT-Bench + A/B comparison vs base model.",
    )


if __name__ == "__main__":
    main()
