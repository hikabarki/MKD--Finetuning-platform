"""Tests for platform/eval_dashboard.py — chart builders."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest


_FULL_DATA = {
    "train_steps": [1, 2, 3],
    "train_loss":  [2.1, 1.8, 1.4],
    "train_ppl":   [8.2, 6.0, 4.1],
    "eval_steps":  [3],
    "eval_loss":   [1.6],
    "eval_ppl":    [4.95],
    "vqa_accuracy_before": 0.35,
    "vqa_accuracy_after":  0.60,
    "vqa_n_samples":       50,
}


def test_build_loss_chart_returns_figure():
    from ft_platform.eval_dashboard import build_loss_chart
    import matplotlib.pyplot as plt
    fig = build_loss_chart(_FULL_DATA)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_build_loss_chart_empty_data():
    from ft_platform.eval_dashboard import build_loss_chart
    import matplotlib.pyplot as plt
    fig = build_loss_chart({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_build_ppl_chart_returns_figure():
    from ft_platform.eval_dashboard import build_ppl_chart
    import matplotlib.pyplot as plt
    fig = build_ppl_chart(_FULL_DATA)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_build_summary_chart_with_vqa():
    from ft_platform.eval_dashboard import build_summary_chart
    import matplotlib.pyplot as plt
    fig = build_summary_chart(_FULL_DATA)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_build_summary_chart_no_data():
    from ft_platform.eval_dashboard import build_summary_chart
    import matplotlib.pyplot as plt
    fig = build_summary_chart({})
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_build_metrics_df_columns():
    from ft_platform.eval_dashboard import build_metrics_df
    df = build_metrics_df(_FULL_DATA)
    assert list(df.columns) == ["Metric", "Before fine-tuning", "After fine-tuning", "Change"]


def test_build_metrics_df_has_rows():
    from ft_platform.eval_dashboard import build_metrics_df
    df = build_metrics_df(_FULL_DATA)
    metrics = df["Metric"].tolist()
    assert any("Loss" in m for m in metrics)
    assert any("VQA" in m for m in metrics)


def test_build_metrics_df_empty():
    from ft_platform.eval_dashboard import build_metrics_df
    df = build_metrics_df({})
    assert len(df) == 0
    assert list(df.columns) == ["Metric", "Before fine-tuning", "After fine-tuning", "Change"]


def test_load_eval_run_no_name():
    from ft_platform.eval_dashboard import load_eval_run
    import matplotlib.pyplot as plt
    loss_fig, ppl_fig, summary_fig, df = load_eval_run(None)
    assert isinstance(loss_fig, plt.Figure)
    assert isinstance(df, pd.DataFrame)
    plt.close("all")


def test_load_eval_run_missing(tmp_project):
    from ft_platform.eval_dashboard import load_eval_run
    import matplotlib.pyplot as plt
    figs = load_eval_run("no_such_run")
    for f in figs[:3]:
        assert isinstance(f, plt.Figure)
    plt.close("all")
