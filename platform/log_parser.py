"""
platform/log_parser.py — Training log parsing and evaluation helpers.
"""
from __future__ import annotations

import ast
import re

import pandas as pd

from inference.state import ROOT


def list_eval_runs() -> list[str]:
    eval_dir = ROOT / "eval_results"
    if not eval_dir.exists():
        return []
    return sorted([p.name for p in eval_dir.iterdir() if p.is_dir()], reverse=True)


def get_loss_df(run_name: str) -> pd.DataFrame:
    if not run_name:
        return pd.DataFrame({"Step": [], "Loss": []})
    p = ROOT / "logs" / f"train_{run_name}.log"
    if not p.exists():
        return pd.DataFrame({"Step": [], "Loss": []})
    steps, losses = [], []
    for line in p.read_text().splitlines():
        m = re.search(r"['\"]loss['\"]\s*[=:]\s*([\d.]+).*?['\"]step['\"]\s*[=:]\s*(\d+)", line)
        if not m:
            m = re.search(r"step['\"]?\s*[=:]\s*(\d+).*?loss['\"]?\s*[=:]\s*([\d.]+)", line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
            continue
        losses.append(float(m.group(1)))
        steps.append(int(m.group(2)))
    if not steps:
        return pd.DataFrame({"Step": [], "Loss": []})
    return pd.DataFrame({"Step": steps, "Loss": losses})


def get_metrics_html(run_name: str) -> str:
    if not run_name:
        return ""
    p = ROOT / "logs" / f"train_{run_name}.log"
    if not p.exists():
        return ""

    text  = p.read_text()
    lines = text.splitlines()

    loss = grad = lr = epoch = ppl = None
    step = 0
    for line in lines:
        if "'loss'" in line and "'eval_loss'" not in line:
            step += 1
            try:
                d     = ast.literal_eval(line.strip())
                loss  = float(d.get("loss",          loss  or 0)) or loss
                grad  = float(d.get("grad_norm",     grad  or 0)) or grad
                lr    = float(d.get("learning_rate", lr    or 0)) or lr
                epoch = float(d.get("epoch",         epoch or 0)) or epoch
                ppl   = float(d.get("ppl",           ppl   or 0)) or ppl
            except Exception:
                pass

    if not step:
        return (
            '<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
            'padding:14px 18px;color:#d29922;font-size:13px;">'
            '⏳ Waiting for first training step…</div>'
        )

    eta_str = elapsed_str = "—"
    total_steps = step
    tqdm_m = re.search(
        r"(\d+)/(\d+)\s*\[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?)", text
    )
    if tqdm_m:
        cur, total_steps = int(tqdm_m.group(1)), int(tqdm_m.group(2))

        def _secs(s: str) -> int:
            parts = list(map(int, s.split(":")))
            return parts[0]*3600 + parts[1]*60 + (parts[2] if len(parts) == 3 else 0)

        elapsed_s   = _secs(tqdm_m.group(3))
        remaining_s = _secs(tqdm_m.group(4))
        eh, em = elapsed_s // 3600,   (elapsed_s % 3600) // 60
        rh, rm = remaining_s // 3600, (remaining_s % 3600) // 60
        elapsed_str = f"{eh}h {em}m" if eh else f"{em}m"
        eta_str     = f"~{rh}h {rm}m" if rh else (f"~{rm}m" if rm else "< 1m")

    pct   = round(step / total_steps * 100) if total_steps else 0
    lc    = "#3fb950" if (loss or 9) < 1.5 else ("#d29922" if (loss or 9) < 3.0 else "#f85149")
    gc    = "#3fb950" if (grad or 9) < 2.0 else ("#d29922" if (grad or 9) < 5.0 else "#f85149")
    bar_c = "#3fb950" if pct > 80 else ("#58a6ff" if pct > 20 else "#d29922")

    def card(label, val, color="#e6edf3"):
        return (
            f'<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;'
            f'padding:12px 16px;flex:1;min-width:110px;">'
            f'<div style="color:#6e7681;font-size:10px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:6px;">{label}</div>'
            f'<div style="color:{color};font-size:22px;font-weight:700;'
            f'font-family:\'SF Mono\',monospace;line-height:1;">{val}</div>'
            f'</div>'
        )

    return f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px 20px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
    <span style="color:#8b949e;font-size:11px;font-weight:600;letter-spacing:0.06em;">
      LIVE METRICS
    </span>
    <span style="color:#8b949e;font-size:11px;">{run_name}</span>
  </div>
  <div style="background:#21262d;border-radius:4px;height:6px;margin-bottom:16px;">
    <div style="background:{bar_c};width:{pct}%;height:6px;border-radius:4px;
                transition:width 0.3s;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:16px;
              font-size:11px;color:#8b949e;">
    <span>Step <b style="color:#c9d1d9;">{step}</b> / {total_steps}</span>
    <span>{pct}% complete</span>
    <span>Elapsed <b style="color:#c9d1d9;">{elapsed_str}</b></span>
    <span>ETA <b style="color:#d29922;">{eta_str}</b></span>
  </div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    {card("Loss",       f"{loss:.4f}"  if loss  else "—", lc)}
    {card("Perplexity", f"{ppl:.3f}"   if ppl   else "—", "#39c5cf")}
    {card("Grad Norm",  f"{grad:.3f}"  if grad  else "—", gc)}
    {card("Epoch",      f"{epoch:.3f}" if epoch else "—", "#e6edf3")}
    {card("Learn Rate", f"{lr:.2e}"    if lr    else "—", "#bc8cff")}
  </div>
</div>"""


def parse_training_log(run_name: str) -> dict:
    log_path = ROOT / "logs" / f"train_{run_name}.log"
    if not log_path.exists():
        return {}
    train_steps, train_loss, train_ppl = [], [], []
    eval_steps,  eval_loss,  eval_ppl  = [], [], []
    step = 0
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            d = ast.literal_eval(line)
        except Exception:
            continue
        if "loss" in d and "eval_loss" not in d and "vqa_accuracy_before" not in d:
            step += 1
            train_steps.append(step)
            train_loss.append(float(d["loss"]))
            train_ppl.append(float(d.get("ppl", 0)) or float(d["loss"]) ** 2.718)
        elif "eval_loss" in d:
            eval_steps.append(step)
            eval_loss.append(float(d["eval_loss"]))
            eval_ppl.append(float(d.get("eval_ppl", 0)) or float(d["eval_loss"]) ** 2.718)
    result = dict(
        train_steps=train_steps, train_loss=train_loss, train_ppl=train_ppl,
        eval_steps=eval_steps,   eval_loss=eval_loss,   eval_ppl=eval_ppl,
    )
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            d = ast.literal_eval(line)
        except Exception:
            continue
        if "vqa_accuracy_before" in d and "vqa_accuracy_after" in d:
            result["vqa_accuracy_before"] = float(d["vqa_accuracy_before"])
            result["vqa_accuracy_after"]  = float(d["vqa_accuracy_after"])
            result["vqa_n_samples"]       = int(d.get("vqa_n_samples", 0))
            break
    return result
