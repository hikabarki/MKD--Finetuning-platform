"""Shared pytest fixtures for the MKD Fine-Tuning Platform test suite."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so all packages resolve
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Point FT_PROJECT at the repo root so inference.state.ROOT is valid
os.environ.setdefault("FT_PROJECT", str(REPO_ROOT))


@pytest.fixture()
def tmp_project(tmp_path: Path, monkeypatch):
    """Isolated project directory with standard subdirectory layout."""
    for sub in ("logs", "eval_results", "datasets", "models", "checkpoints", "exports"):
        (tmp_path / sub).mkdir()

    import inference.state as _state
    import ft_platform.log_parser as _lp

    monkeypatch.setattr(_state, "ROOT", tmp_path)
    monkeypatch.setattr(_lp,    "ROOT", tmp_path)
    return tmp_path


@pytest.fixture()
def sample_log(tmp_project: Path) -> Path:
    """Write a minimal training log with 3 steps and 1 eval record."""
    log = tmp_project / "logs" / "train_run_test.log"
    log.write_text("\n".join([
        "{'loss': 2.1, 'grad_norm': 1.5, 'learning_rate': 0.0002, 'epoch': 0.1, 'step': 1}",
        "{'loss': 1.8, 'grad_norm': 1.2, 'learning_rate': 0.0002, 'epoch': 0.2, 'step': 2}",
        "{'loss': 1.4, 'grad_norm': 0.9, 'learning_rate': 0.0002, 'epoch': 0.3, 'step': 3}",
        "{'eval_loss': 1.6, 'eval_ppl': 4.95}",
        "{'vqa_accuracy_before': 0.35, 'vqa_accuracy_after': 0.60, 'vqa_n_samples': 50}",
    ]))
    return log
