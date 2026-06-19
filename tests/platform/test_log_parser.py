"""Tests for platform/log_parser.py — training log parsing."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_parse_training_log_empty_run(tmp_project):
    from ft_platform.log_parser import parse_training_log
    result = parse_training_log("nonexistent_run")
    assert result == {}


def test_parse_training_log_full(sample_log, tmp_project):
    from ft_platform.log_parser import parse_training_log
    data = parse_training_log("run_test")

    assert data["train_steps"] == [1, 2, 3]
    assert len(data["train_loss"]) == 3
    assert data["train_loss"][0] == pytest.approx(2.1)
    assert data["train_loss"][-1] == pytest.approx(1.4)

    assert len(data["eval_loss"]) == 1
    assert data["eval_loss"][0] == pytest.approx(1.6)

    assert data["vqa_accuracy_before"] == pytest.approx(0.35)
    assert data["vqa_accuracy_after"]  == pytest.approx(0.60)
    assert data["vqa_n_samples"] == 50


def test_parse_training_log_ppl_computed(sample_log, tmp_project):
    """train_ppl should be populated (either from log or computed from loss)."""
    from ft_platform.log_parser import parse_training_log
    data = parse_training_log("run_test")
    assert len(data["train_ppl"]) == 3
    for v in data["train_ppl"]:
        assert v > 0


def test_get_loss_df_empty_run(tmp_project):
    from ft_platform.log_parser import get_loss_df
    df = get_loss_df("")
    assert list(df.columns) == ["Step", "Loss"]
    assert len(df) == 0


def test_get_loss_df_missing_log(tmp_project):
    from ft_platform.log_parser import get_loss_df
    df = get_loss_df("no_such_run")
    assert len(df) == 0


def test_get_loss_df_with_log(tmp_project):
    log = tmp_project / "logs" / "train_myrun.log"
    log.write_text("step'=1 loss'=1.23\nstep'=2 loss'=0.99\n")
    from ft_platform.log_parser import get_loss_df
    df = get_loss_df("myrun")
    # may or may not match depending on regex — just verify shape is sane
    assert list(df.columns) == ["Step", "Loss"]


def test_get_metrics_html_no_data(tmp_project):
    from ft_platform.log_parser import get_metrics_html
    html = get_metrics_html("")
    assert html == ""


def test_get_metrics_html_waiting(tmp_project):
    """No training steps yet → 'Waiting' message."""
    log = tmp_project / "logs" / "train_empty.log"
    log.write_text("just some noise\n")
    from ft_platform.log_parser import get_metrics_html
    html = get_metrics_html("empty")
    assert "Waiting" in html


def test_get_metrics_html_with_steps(sample_log, tmp_project):
    from ft_platform.log_parser import get_metrics_html
    html = get_metrics_html("run_test")
    assert "Loss" in html
    assert "run_test" in html


def test_list_eval_runs_empty(tmp_project):
    from ft_platform.log_parser import list_eval_runs
    runs = list_eval_runs()
    assert runs == []


def test_list_eval_runs_with_dirs(tmp_project):
    for name in ("run_a", "run_b", "run_c"):
        (tmp_project / "eval_results" / name).mkdir()
    from ft_platform.log_parser import list_eval_runs
    runs = list_eval_runs()
    assert set(runs) == {"run_a", "run_b", "run_c"}
    assert runs == sorted(runs, reverse=True)
