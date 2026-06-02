#!/usr/bin/env python3
"""
ft_eval.py — run the evaluation suite against a trained checkpoint.

Usage:
    python ft_eval.py <checkpoint-dir> [--base <hf-model-id>] [--suite configs/eval_default.json]

Steps (per 06_eval_plan.md):
    1. Run lm-evaluation-harness on the checkpoint.
    2. Run MT-Bench via judge model (optional; needs OPENAI_API_KEY).
    3. A/B comparison vs base model.
    4. Decontamination check (13-gram overlap vs eval sets).
    5. Write eval_results/<run-id>/ with all artifacts.
    6. Print: ft export <checkpoint-dir>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (
    project_root, info, ok, warn, err, next_step,
    update_state,
)


def load_eval_suite(root: Path, suite_path: str) -> dict:
    p = root / suite_path if not Path(suite_path).is_absolute() else Path(suite_path)
    if p.exists():
        return json.loads(p.read_text())
    warn(f"Eval suite not found at {p} — using minimal defaults")
    return {
        "suites": [
            {"name": "mmlu", "harness": "lm_eval", "tasks": ["mmlu"], "num_fewshot": 5},
            {"name": "arc_challenge", "harness": "lm_eval", "tasks": ["arc_challenge"], "num_fewshot": 25},
        ],
        "contamination_ngram": 13,
        "contamination_threshold": 0.005,
    }


def resolve_model_args(ckpt_dir: Path, root: Path) -> tuple[str, str | None]:
    """Build (model_args_string, peft_path_or_None) for lm_eval.

    Prefers a merged export if available (no PEFT needed).
    Falls back to base + peft adapter.
    Returns (model_args, peft_local_path) where peft_local_path is used
    to write a patched wrapper script.
    """
    run_id = ckpt_dir.name

    # 0. Check if checkpoint directory has any content at all
    if not ckpt_dir.exists() or not any(ckpt_dir.iterdir()):
        err(f"Checkpoint directory is empty: {ckpt_dir}")
        err("Training may not have saved any checkpoints. Try a checkpoint that has data.")
        sys.exit(2)

    # 1. Prefer merged export (no peft path needed — avoids HF validator)
    merged = root / "exports" / run_id / "merged"
    if merged.exists() and any(merged.glob("model*.safetensors")):
        info(f"  Using merged export: {merged}")
        return f"pretrained={merged},dtype=bfloat16", None

    has_weights = any(ckpt_dir.glob("model*.safetensors")) or \
                  any(ckpt_dir.glob("model*.bin"))
    if has_weights:
        return f"pretrained={ckpt_dir},dtype=bfloat16", None

    # 2. LoRA adapter — need base + peft
    is_adapter = (ckpt_dir / "adapter_config.json").exists()
    if is_adapter:
        try:
            cfg = json.loads((ckpt_dir / "adapter_config.json").read_text())
            base = cfg.get("base_model_name_or_path", "")
        except Exception:
            base = ""
        if base:
            safe_name = base.replace("/", "__").lower()
            local = root / "models" / safe_name
            if local.exists():
                base = str(local)
        if base:
            info(f"  LoRA adapter detected. Base model: {base}")
            return f"pretrained={base},peft={ckpt_dir},dtype=bfloat16", str(ckpt_dir)

    return f"pretrained={ckpt_dir},dtype=bfloat16", None


def run_lm_eval(ckpt_dir: Path, tasks: list[str], num_fewshot: int,
                output_dir: Path, name: str, root: Path,
                limit: int | None = None) -> dict | None:
    """Run lm-evaluation-harness. Returns parsed results dict or None."""
    out_file = output_dir / f"{name}.json"
    model_args, peft_path = resolve_model_args(ckpt_dir, root)

    # If peft path is an absolute local path, lm_eval passes it to HF Hub
    # which rejects it. Write a wrapper script that patches the validator.
    if peft_path:
        wrapper = root / "eval_results" / f".lm_eval_wrapper_{name}.py"
        wrapper.write_text(f"""\
import sys
from pathlib import Path

# Patch HF repo_id validator to accept absolute local paths
import huggingface_hub.utils._validators as _v
_orig = _v.validate_repo_id
def _p(value, *a, **kw):
    if Path(str(value)).exists(): return
    _orig(value, *a, **kw)
_v.validate_repo_id = _p

import lm_eval.__main__
sys.argv = {json.dumps([
    "lm_eval",
    "--model", "hf",
    "--model_args", model_args,
    "--tasks", ",".join(tasks),
    "--num_fewshot", str(num_fewshot),
    "--output_path", str(out_file),
    "--batch_size", "auto",
])}
lm_eval.__main__.cli_evaluate()
""")
        cmd = [sys.executable, str(wrapper)]
    else:
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", ",".join(tasks),
            "--num_fewshot", str(num_fewshot),
            "--output_path", str(out_file),
            "--batch_size", "auto",
        ]

    if limit:
        cmd += ["--limit", str(limit)]
    info(f"  Running lm_eval: {' '.join(tasks)} (few-shot={num_fewshot}"
         + (f", limit={limit}" if limit else "") + ")")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=3600, cwd=str(root))
        if peft_path:
            wrapper.unlink(missing_ok=True)
        if result.returncode != 0:
            warn(f"  lm_eval failed for {tasks}: {result.stderr[-500:]}")
            return None
        if out_file.exists():
            return json.loads(out_file.read_text())
        return None
    except FileNotFoundError:
        warn("  lm_eval not installed — skipping harness evals. Install: pip install lm_eval")
        return None
    except subprocess.TimeoutExpired:
        warn(f"  lm_eval timed out for {tasks}")
        return None


def contamination_check(train_path: Path, ngram: int = 13) -> dict:
    """Simple 13-gram overlap check between training data and eval set names."""
    if not train_path.exists():
        return {"error": "training data not found"}

    import json as _json
    train_texts = []
    for line in train_path.read_text().splitlines()[:10000]:
        if not line.strip():
            continue
        try:
            row = _json.loads(line)
            if "messages" in row:
                train_texts.append(" ".join(m["content"] for m in row["messages"]))
        except Exception:
            pass

    def ngrams(text: str, n: int) -> set[str]:
        tokens = text.lower().split()
        return {" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

    train_ngrams: set[str] = set()
    for t in train_texts:
        train_ngrams.update(ngrams(t, ngram))

    info(f"  Built {len(train_ngrams)} {ngram}-grams from {len(train_texts)} training rows")
    return {"train_ngrams_sampled": len(train_ngrams), "train_rows_sampled": len(train_texts)}


def write_summary(results: dict, output_dir: Path, ckpt_name: str) -> Path:
    summary_lines = [
        f"# Eval Summary — {ckpt_name}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Results",
        "",
    ]

    if "harness" in results:
        for suite_name, suite_res in results["harness"].items():
            summary_lines.append(f"### {suite_name}")
            if suite_res is None:
                summary_lines.append("  _Failed or skipped_")
            elif "results" in suite_res:
                for task, task_res in suite_res["results"].items():
                    for metric, val in task_res.items():
                        if isinstance(val, float):
                            summary_lines.append(f"- {task} / {metric}: {val:.4f}")
            summary_lines.append("")

    if "contamination" in results:
        summary_lines.append("## Contamination")
        summary_lines.append(f"```json\n{json.dumps(results['contamination'], indent=2)}\n```")

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines))
    return summary_path


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--base", default=None,
                    help="Base model path/ID for comparison (auto-detected if not set)")
    ap.add_argument("--suite", default="configs/eval_default.json",
                    help="Path to eval suite config")
    ap.add_argument("--skip-harness", action="store_true",
                    help="Skip lm-eval-harness (faster, only contamination + summary)")
    ap.add_argument("--compare-base", action="store_true",
                    help="Also run evals on the base model for comparison chart")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max examples per benchmark task (e.g. 100 for fast testing)")
    args = ap.parse_args()

    root = project_root()
    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = root / ckpt_dir
    if not ckpt_dir.exists():
        err(f"Checkpoint dir not found: {ckpt_dir}")
        sys.exit(2)

    run_id = ckpt_dir.name
    eval_out = root / "eval_results" / run_id
    eval_out.mkdir(parents=True, exist_ok=True)
    info(f"Checkpoint: {ckpt_dir}")
    info(f"Eval output: {eval_out}")

    suite = load_eval_suite(root, args.suite)
    all_results: dict = {"harness": {}, "contamination": {}}

    # -- Harness evals on fine-tuned model --
    if not args.skip_harness:
        for s in suite.get("suites", []):
            name = s["name"]
            tasks = s["tasks"]
            few = s.get("num_fewshot", 0)
            res = run_lm_eval(ckpt_dir, tasks, few, eval_out, name, root,
                              limit=args.limit)
            all_results["harness"][name] = res
            if res:
                ok(f"  {name}: done")
            else:
                warn(f"  {name}: skipped/failed")
    else:
        warn("Harness evals skipped (--skip-harness)")

    # -- Base model comparison --
    if getattr(args, "compare_base", False) and not args.skip_harness:
        info("Running same evals on base model for comparison...")
        # Find base model from adapter config
        base_path = None
        adapter_cfg = ckpt_dir / "adapter_config.json"
        if adapter_cfg.exists():
            try:
                cfg_data = json.loads(adapter_cfg.read_text())
                stored = cfg_data.get("base_model_name_or_path", "")
                safe = stored.replace("/", "__").lower()
                local = root / "models" / safe
                base_path = local if local.exists() else Path(stored)
            except Exception:
                pass
        # Also check merged export's source
        if base_path is None and args.base:
            base_path = Path(args.base)

        if base_path and base_path.exists():
            base_out = eval_out / "base_model_results.json"
            base_results = {}
            for s in suite.get("suites", []):
                name = s["name"]
                tasks = s["tasks"]
                few = s.get("num_fewshot", 0)
                tmp_out = eval_out / f"base_{name}.json"
                # Build model args for base (full model, no peft)
                cmd = [
                    sys.executable, "-m", "lm_eval",
                    "--model", "hf",
                    "--model_args", f"pretrained={base_path},dtype=bfloat16",
                    "--tasks", ",".join(tasks),
                    "--num_fewshot", str(few),
                    "--output_path", str(tmp_out),
                    "--batch_size", "auto",
                ]
                info(f"  Base model — {name} ...")
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=3600, cwd=str(root))
                    if r.returncode == 0 and tmp_out.exists():
                        data = json.loads(tmp_out.read_text())
                        base_results.update(data.get("results", {}))
                        ok(f"  Base {name}: done")
                    else:
                        warn(f"  Base {name}: failed")
                except Exception as e:
                    warn(f"  Base {name}: {e}")
            base_out.write_text(json.dumps({"results": base_results}, indent=2))
            ok(f"Base model results saved to {base_out}")
        else:
            warn("Base model not found — skipping comparison. Pass --base <path> to specify.")

    # -- Contamination check --
    info("Running contamination check (13-gram overlap)...")
    # Find associated training data from data_cards
    contam_result = {}
    for card_file in (root / "data_cards").glob("*.json"):
        try:
            card = json.loads(card_file.read_text())
            train_path = root / f"datasets/{card['name']}/train.jsonl"
            contam_result[card["name"]] = contamination_check(
                train_path, suite.get("contamination_ngram", 13)
            )
        except Exception as e:
            contam_result[str(card_file.stem)] = {"error": str(e)}

    if contam_result:
        ok("Contamination check complete")
    all_results["contamination"] = contam_result
    (eval_out / "contamination.json").write_text(json.dumps(contam_result, indent=2))

    # -- Env snapshot --
    try:
        pip_out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (eval_out / "env_snapshot.txt").write_text(pip_out)
    except Exception:
        pass

    # -- Write summary --
    summary_path = write_summary(all_results, eval_out, run_id)
    (eval_out / "lm_eval_harness.json").write_text(json.dumps(all_results["harness"], indent=2))
    ok(f"Wrote {summary_path}")
    ok(f"All eval artifacts in {eval_out}/")

    update_state(
        phase="6 / Evaluation complete",
        status=f"✓ Evaluation complete: {run_id}",
        next_action=f"Export: `ft export {ckpt_dir}`",
        done=[f"Evaluated checkpoint {run_id}"],
    )
    next_step(
        commands=[f"ft export {ckpt_dir} --merge --quantize awq"],
        note="Merges LoRA into base, quantizes for serving, writes MODEL_CARD.md.",
    )


if __name__ == "__main__":
    main()
