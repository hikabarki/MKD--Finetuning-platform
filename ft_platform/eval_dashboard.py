"""
platform/eval_dashboard.py — Matplotlib chart builders for the eval dashboard.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from ft_platform.log_parser import parse_training_log

# ── Chart palette ─────────────────────────────────────────────────────────────
_FIG_BG    = "#0d1117"
_AXES_BG   = "#161b22"
_GRID_COL  = "#30363d"
_TICK_COL  = "#c9d1d9"
_LABEL_COL = "#8b949e"
_TITLE_COL = "#e6edf3"
_PURPLE    = "#a371f7"
_CYAN      = "#39c5cf"
_GREY_LINE = "#6e7681"
_GREY_BAR  = "#484f58"
_CHART_W   = 4.8
_CHART_H   = 3.6
_DPI       = 100


def _new_fig():
    fig, ax = plt.subplots(figsize=(_CHART_W, _CHART_H), dpi=_DPI)
    fig.patch.set_facecolor(_FIG_BG)
    ax.set_facecolor(_AXES_BG)
    return fig, ax


def _style_ax(ax, title: str, xlabel: str, ylabel: str, legend: bool = True):
    ax.set_title(title, color=_TITLE_COL, fontsize=11, fontweight="bold",
                 pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=_LABEL_COL, fontsize=9.5, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, color=_LABEL_COL, fontsize=9.5, labelpad=6)
    ax.tick_params(axis="both", which="both",
                   colors=_TICK_COL, labelcolor=_TICK_COL,
                   labelsize=8.5, length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COL)
        spine.set_linewidth(0.7)
    ax.grid(axis="both", color=_GRID_COL, linestyle="--",
            linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    if legend:
        leg = ax.legend(facecolor="#21262d", edgecolor=_GRID_COL,
                        fontsize=8.5, framealpha=0.95, loc="upper right",
                        handlelength=1.6)
        for t in leg.get_texts():
            t.set_color(_TITLE_COL)


def _annotate_endpoints(ax, steps, vals, color):
    ax.annotate(f"{vals[0]:.3f}",
                xy=(steps[0], vals[0]), xytext=(5, 5),
                textcoords="offset points",
                color=_TICK_COL, fontsize=7.5, ha="left")
    ax.annotate(f"{vals[-1]:.3f}",
                xy=(steps[-1], vals[-1]), xytext=(-5, 5),
                textcoords="offset points",
                color=color, fontsize=7.5, fontweight="bold", ha="right")


def build_loss_chart(data: dict):
    fig, ax = _new_fig()
    if data.get("train_steps"):
        steps, loss = data["train_steps"], data["train_loss"]
        ax.plot(steps, loss, color=_PURPLE, linewidth=2.0,
                marker="o", markersize=3, zorder=3, label="Fine-tuned model")
        ax.axhline(loss[0], color=_GREY_LINE, linestyle="--",
                   linewidth=1.2, zorder=2, label="Base model")
        ax.fill_between(steps, loss, loss[0],
                        where=[v < loss[0] for v in loss],
                        color=_PURPLE, alpha=0.10, zorder=1)
        _annotate_endpoints(ax, steps, loss, _PURPLE)
    else:
        ax.text(0.5, 0.5, "No training data yet", ha="center", va="center",
                color=_LABEL_COL, transform=ax.transAxes, fontsize=11)
    _style_ax(ax, "Training Loss", "Training Steps", "Loss")
    fig.tight_layout(pad=1.4)
    return fig


def build_ppl_chart(data: dict):
    fig, ax = _new_fig()
    if data.get("train_steps"):
        steps, ppl = data["train_steps"], data["train_ppl"]
        ax.plot(steps, ppl, color=_CYAN, linewidth=2.0,
                marker="o", markersize=3, zorder=3, label="Fine-tuned model")
        ax.axhline(ppl[0], color=_GREY_LINE, linestyle="--",
                   linewidth=1.2, zorder=2, label="Base model")
        ax.fill_between(steps, ppl, ppl[0],
                        where=[v < ppl[0] for v in ppl],
                        color=_CYAN, alpha=0.10, zorder=1)
        _annotate_endpoints(ax, steps, ppl, _CYAN)
    else:
        ax.text(0.5, 0.5, "No training data yet", ha="center", va="center",
                color=_LABEL_COL, transform=ax.transAxes, fontsize=11)
    _style_ax(ax, "Perplexity", "Training Steps", "Perplexity")
    fig.tight_layout(pad=1.4)
    return fig


def build_summary_chart(data: dict):
    fig, ax = _new_fig()
    has_vqa = "vqa_accuracy_before" in data and "vqa_accuracy_after" in data
    if data.get("train_loss"):
        bl = data["train_loss"][0];  al = data["train_loss"][-1]
        bp = data["train_ppl"][0];   ap = data["train_ppl"][-1]
        if data.get("eval_loss") and len(data["eval_loss"]) >= 2:
            al = data["eval_loss"][-1]
        if data.get("eval_ppl") and len(data["eval_ppl"]) >= 2:
            ap = data["eval_ppl"][-1]
        x_pos        = [0, 1]
        x_labels     = ["Loss", "Perplexity"]
        before_vals  = [bl, bp]
        after_vals   = [al, ap]
        if has_vqa:
            x_pos.append(2)
            x_labels.append("VQA Accuracy")
            before_vals.append(data["vqa_accuracy_before"])
            after_vals.append(data["vqa_accuracy_after"])
        w  = 0.30
        b1 = ax.bar([i - w/2 for i in x_pos], before_vals, w,
                    color=_GREY_BAR, label="Before", zorder=3,
                    edgecolor=_GRID_COL, linewidth=0.5)
        b2 = ax.bar([i + w/2 for i in x_pos], after_vals, w,
                    color=_PURPLE,   label="After",  zorder=3,
                    edgecolor=_GRID_COL, linewidth=0.5)
        max_y = max(before_vals + after_vals)
        for bar in list(b1) + list(b2):
            h     = bar.get_height()
            label = f"{h:.1%}" if has_vqa and bar.get_x() > 1.5 else f"{h:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + max_y * 0.015, label,
                    ha="center", va="bottom",
                    color=_TITLE_COL, fontsize=8, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, color=_TICK_COL, fontsize=9.5)
        ax.set_ylim(0, max_y * 1.28)
    elif has_vqa:
        vb = data["vqa_accuracy_before"]; va = data["vqa_accuracy_after"]
        b1 = ax.bar([0 - 0.15], [vb], 0.30, color=_GREY_BAR, label="Before", zorder=3)
        b2 = ax.bar([0 + 0.15], [va], 0.30, color=_PURPLE,   label="After",  zorder=3)
        for bar, val in zip(list(b1)+list(b2), [vb, va]):
            ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.1%}",
                    ha="center", va="bottom", color=_TITLE_COL, fontsize=9, fontweight="bold")
        ax.set_xticks([0])
        ax.set_xticklabels(["VQA Accuracy"], color=_TICK_COL, fontsize=9.5)
        ax.set_ylim(0, max(vb, va) * 1.3)
    else:
        ax.text(0.5, 0.5, "No data yet", ha="center", va="center",
                color=_LABEL_COL, transform=ax.transAxes, fontsize=11)
    _style_ax(ax, "Before vs After", "", "Value")
    fig.tight_layout(pad=1.4)
    return fig


def build_metrics_df(data: dict) -> pd.DataFrame:
    cols = ["Metric", "Before fine-tuning", "After fine-tuning", "Change"]
    if not data.get("train_loss"):
        return pd.DataFrame(columns=cols)
    bl = data["train_loss"][0];  al = data["train_loss"][-1]
    bp = data["train_ppl"][0];   ap = data["train_ppl"][-1]
    def _pct(before, after):
        if before == 0:
            return "N/A"
        return f"{(after - before) / before * 100:+.1f}%"
    rows = [
        ("Train Loss",       f"{bl:.4f}", f"{al:.4f}", _pct(bl, al)),
        ("Train Perplexity", f"{bp:.3f}", f"{ap:.3f}", _pct(bp, ap)),
    ]
    if data.get("eval_loss") and len(data["eval_loss"]) >= 2:
        el_b = data["eval_loss"][0]; el_a = data["eval_loss"][-1]
        rows.append(("Eval Loss", f"{el_b:.4f}", f"{el_a:.4f}", _pct(el_b, el_a)))
    if data.get("eval_ppl") and len(data["eval_ppl"]) >= 2:
        ep_b = data["eval_ppl"][0]; ep_a = data["eval_ppl"][-1]
        rows.append(("Eval Perplexity", f"{ep_b:.3f}", f"{ep_a:.3f}", _pct(ep_b, ep_a)))
    if "vqa_accuracy_before" in data and "vqa_accuracy_after" in data:
        vb  = data["vqa_accuracy_before"]; va = data["vqa_accuracy_after"]
        n   = data.get("vqa_n_samples", "?")
        chg = f"{(va - vb) * 100:+.1f}pp"
        rows.append((f"VQA Accuracy  (n={n})", f"{vb:.1%}", f"{va:.1%}", chg))
    return pd.DataFrame(rows, columns=cols)


def load_eval_run(run_name: str):
    empty = pd.DataFrame(columns=["Metric", "Before fine-tuning",
                                   "After fine-tuning", "Change"])
    if not run_name:
        return build_loss_chart({}), build_ppl_chart({}), build_summary_chart({}), empty
    data = parse_training_log(run_name)
    return (build_loss_chart(data), build_ppl_chart(data),
            build_summary_chart(data), build_metrics_df(data))
