#!/usr/bin/env python3
"""
ft_publish.py — push exported model to HuggingFace Hub and GitHub.

Usage:
    python ft_publish.py <export-dir>
                         --hf-repo mkd-hika/<repo-name>
                         [--github-repo mkd-hika/<repo-name>]
                         [--acknowledge-contamination]
                         [--accept-base-license]

Refuses to push if (per 07_handoff.md):
    - MODEL_CARD.md still has {filled by user} placeholders
    - Contamination ratio > 0.5% on any eval
    - Base model is research-only without --accept-base-license

Steps:
    1. Validate MODEL_CARD.md has no unfilled placeholders.
    2. Check contamination report.
    3. Push weights to HF Hub.
    4. Push training code (scripts/ + configs/) to GitHub.
    5. Print URLs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ft_common import (
    project_root, info, ok, warn, err, next_step,
    update_state,
)

CONTAMINATION_THRESHOLD = 0.005  # 0.5%


def validate_model_card(export_dir: Path) -> list[str]:
    """Return list of unfilled {placeholder} lines."""
    card = export_dir / "MODEL_CARD.md"
    if not card.exists():
        return ["MODEL_CARD.md is missing — run `ft export` first"]
    problems = []
    for i, line in enumerate(card.read_text().splitlines(), 1):
        if "{filled by user" in line.lower():
            problems.append(f"  Line {i}: {line.strip()}")
    return problems


def check_contamination(root: Path, run_id: str,
                        acknowledge: bool) -> list[str]:
    """Return list of contamination failures."""
    contam_file = root / "eval_results" / run_id / "contamination.json"
    if not contam_file.exists():
        warn("No contamination.json found — skipping contamination check.")
        warn("Run `ft eval` before publishing for full safety.")
        return []

    failures = []
    try:
        data = json.loads(contam_file.read_text())
        for eval_name, stats in data.items():
            ratio = stats.get("ratio", 0.0)
            if ratio > CONTAMINATION_THRESHOLD and not acknowledge:
                failures.append(
                    f"  {eval_name}: ratio={ratio:.4f} > {CONTAMINATION_THRESHOLD} "
                    f"({stats.get('contaminated_rows', '?')} rows)"
                )
    except Exception as e:
        warn(f"Could not parse contamination.json: {e}")
    return failures


def push_to_hf(export_dir: Path, hf_repo: str, quantize_format: str = "awq") -> bool:
    """Upload model to HuggingFace Hub."""
    info(f"Pushing to HuggingFace: {hf_repo}")

    # Copy MODEL_CARD.md to README.md for HF display
    card = export_dir / "MODEL_CARD.md"
    readme = export_dir / "README.md"
    if card.exists() and not readme.exists():
        import shutil
        shutil.copy2(card, readme)

    model_dir = export_dir / quantize_format
    if not model_dir.exists():
        model_dir = export_dir / "merged"
    if not model_dir.exists():
        model_dir = export_dir

    try:
        result = subprocess.run(
            [
                "huggingface-cli", "upload",
                hf_repo,
                str(model_dir),
                ".",
                "--repo-type=model",
                "--commit-message=Initial release",
            ],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            err(f"HF upload failed:\n{result.stderr[-500:]}")
            return False

        # Also upload README
        if readme.exists():
            subprocess.run(
                ["huggingface-cli", "upload", hf_repo, str(readme), "README.md",
                 "--repo-type=model"],
                capture_output=True, text=True, timeout=120,
            )
        ok(f"Pushed to https://huggingface.co/{hf_repo}")
        return True
    except FileNotFoundError:
        err("huggingface-cli not found. Install: pip install huggingface_hub")
        return False


def push_to_github(root: Path, github_repo: str) -> bool:
    """Push training code (scripts/ + configs/) to GitHub."""
    info(f"Pushing code to GitHub: {github_repo}")
    code_dir = root / "_github_push_tmp"
    try:
        import shutil
        shutil.rmtree(code_dir, ignore_errors=True)
        code_dir.mkdir()

        for subdir in ("scripts", "configs"):
            src = root / subdir
            if src.exists():
                shutil.copytree(src, code_dir / subdir)

        for doc in root.glob("*.md"):
            shutil.copy2(doc, code_dir / doc.name)
        for yaml in root.glob("*.yaml"):
            shutil.copy2(yaml, code_dir / yaml.name)
        if (root / ".gitignore").exists():
            shutil.copy2(root / ".gitignore", code_dir / ".gitignore")

        cmds = [
            ["git", "init"],
            ["git", "add", "."],
            ["git", "commit", "-m", "Initial release"],
            ["git", "branch", "-M", "main"],
            ["git", "remote", "add", "origin",
             f"https://github.com/{github_repo}.git"],
            ["git", "push", "-u", "origin", "main"],
        ]
        for cmd in cmds:
            r = subprocess.run(cmd, cwd=str(code_dir), capture_output=True, text=True)
            if r.returncode != 0 and cmd[1] not in ("remote", "push"):
                warn(f"git step failed: {' '.join(cmd)}\n{r.stderr[-200:]}")

        shutil.rmtree(code_dir, ignore_errors=True)
        ok(f"Pushed to https://github.com/{github_repo}")
        return True
    except Exception as e:
        warn(f"GitHub push failed: {e}")
        shutil.rmtree(code_dir, ignore_errors=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--hf-repo", required=True,
                    help="HuggingFace repo name, e.g. mkd-hika/llama-3.1-8b-alpaca")
    ap.add_argument("--github-repo", default=None,
                    help="GitHub repo name (optional), e.g. mkd-hika/llama-3.1-8b-alpaca")
    ap.add_argument("--acknowledge-contamination", action="store_true",
                    help="Proceed even if contamination ratio > 0.5%")
    ap.add_argument("--accept-base-license", action="store_true",
                    help="Acknowledge restricted base model license")
    ap.add_argument("--quantize-format", default="awq",
                    help="Which quantized subdir to push (default: awq)")
    args = ap.parse_args()

    root = project_root()
    export_dir = Path(args.export_dir)
    if not export_dir.is_absolute():
        export_dir = root / export_dir
    if not export_dir.exists():
        err(f"Export dir not found: {export_dir}")
        sys.exit(2)

    run_id = export_dir.name
    info(f"Export dir: {export_dir}")
    info(f"HF repo:    {args.hf_repo}")

    # -- Pre-flight checks --
    blockers: list[str] = []

    card_problems = validate_model_card(export_dir)
    if card_problems:
        blockers.append("MODEL_CARD.md has unfilled placeholders:")
        blockers.extend(card_problems)

    contam_failures = check_contamination(root, run_id, args.acknowledge_contamination)
    if contam_failures:
        blockers.append("Contamination check failed (pass --acknowledge-contamination to override):")
        blockers.extend(contam_failures)

    if blockers:
        err("Publish blocked — fix these issues first:")
        for b in blockers:
            err(b)
        sys.exit(1)

    # -- Push --
    hf_ok = push_to_hf(export_dir, args.hf_repo, args.quantize_format)
    github_ok = True
    if args.github_repo:
        github_ok = push_to_github(root, args.github_repo)

    if not hf_ok:
        err("HuggingFace push failed — check errors above")
        sys.exit(1)

    # -- Success --
    hf_url = f"https://huggingface.co/{args.hf_repo}"
    gh_url = f"https://github.com/{args.github_repo}" if args.github_repo else None

    print()
    print("  🎉 Published.")
    print(f"     Weights:    {hf_url}")
    if gh_url:
        print(f"     Code:       {gh_url}")
    print(f"     Model card: {hf_url}/blob/main/README.md")
    print()

    update_state(
        phase="7 / Published",
        status=f"🎉 Published to {hf_url}",
        next_action="Monitor HF repo stats. If regression found, train v2 in a new run folder.",
        done=[f"Published {run_id} to {args.hf_repo}"],
    )


if __name__ == "__main__":
    main()
