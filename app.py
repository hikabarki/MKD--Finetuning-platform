#!/usr/bin/env python3
"""
app.py — Production-grade Gradio UI for the MKD Fine-Tuning Platform.

Launch:  ft ui  |  python app.py  |  python app.py --port 7860 --share
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import ast

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Chat model state (loaded lazily) ─────────────────────────────────────────
_chat_model  = None
_chat_tok    = None
_chat_name   = None

# ── Project root ──────────────────────────────────────────────────────────────
ROOT   = Path(os.environ.get("FT_PROJECT", Path(__file__).resolve().parent))
SCRIPTS = ROOT / "scripts"
PYTHON  = sys.executable

# ── ANSI → HTML terminal renderer ─────────────────────────────────────────────
_ANSI_COLOR = {
    "32": "#3fb950", "33": "#d29922", "31": "#f85149",
    "36": "#39c5cf", "35": "#bc8cff", "34": "#58a6ff",
    "2":  "#8b949e", "1":  None,
}

def ansi_to_html(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    open_span = False

    def replace(m):
        nonlocal open_span
        codes = m.group(1).split(";")
        if codes == ["0"] or codes == [""]:
            end = "</span>" if open_span else ""
            open_span = False
            return end
        styles, attrs = [], []
        for c in codes:
            if c in _ANSI_COLOR:
                v = _ANSI_COLOR[c]
                if v:
                    styles.append(f"color:{v}")
                else:
                    attrs.append("font-weight:bold")
        style = ";".join(styles + attrs)
        if style:
            if open_span:
                result = f"</span><span style='{style}'>"
            else:
                result = f"<span style='{style}'>"
            open_span = True
            return result
        return ""

    html = re.sub(r"\033\[([0-9;]*)m", replace, text)
    html = re.sub(r"\033\[[0-9]*[A-HJKST]", "", html)
    if open_span:
        html += "</span>"
    html = html.replace("\n", "<br>")
    # Strip box-drawing chars that don't render cleanly in HTML
    for ch, rep in [("╭","┌"),("╮","┐"),("╰","└"),("╯","┘"),("─","─"),("│","│")]:
        pass  # keep them, they render fine
    return html


def terminal_wrap(content: str, height: int = 380) -> str:
    return f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
            font-family:'JetBrains Mono',Consolas,monospace;font-size:13px;
            color:#e6edf3;line-height:1.65;padding:16px;
            max-height:{height}px;overflow-y:auto;white-space:pre-wrap;">
{content if content else '<span style="color:#8b949e">Output will appear here...</span>'}
</div>"""


# ── Data helpers ───────────────────────────────────────────────────────────────
def list_datasets()         -> list[str]:
    d = ROOT / "datasets"
    return [p.name for p in d.iterdir() if p.is_dir()] if d.exists() else []

def list_cleaned_datasets() -> list[str]:
    d = ROOT / "datasets"
    return [p.name for p in d.iterdir()
            if p.is_dir() and (p / "train.jsonl").exists()] if d.exists() else []

def list_models()           -> list[str]:
    d = ROOT / "models"
    return [p.name for p in d.iterdir() if p.is_dir()] if d.exists() else []

def list_configs()          -> list[str]:
    d = ROOT / "configs"
    return sorted([p.name for p in d.glob("run_*.yaml")], reverse=True) if d.exists() else []

def list_checkpoints()      -> list[str]:
    d = ROOT / "checkpoints"
    return sorted([p.name for p in d.iterdir() if p.is_dir()], reverse=True) if d.exists() else []

def list_exports()          -> list[str]:
    d = ROOT / "exports"
    return sorted([p.name for p in d.iterdir() if p.is_dir()], reverse=True) if d.exists() else []


def get_gpu_html() -> str:
    try:
        rows = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=4,
        ).strip().splitlines()
        cards = []
        for row in rows:
            idx, name, util, mu, mt, temp, pwr = [p.strip() for p in row.split(",")]
            util_i = int(util)
            bar_w  = util_i
            bar_col = "#3fb950" if util_i < 70 else ("#d29922" if util_i < 90 else "#f85149")
            cards.append(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 18px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <span style="color:#e6edf3;font-weight:600;font-size:14px;">GPU {idx} · {name}</span>
    <span style="color:#8b949e;font-size:12px;">{temp}°C · {pwr}W</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <span style="color:#8b949e;font-size:12px;width:36px;">Util</span>
    <div style="flex:1;background:#21262d;border-radius:4px;height:8px;">
      <div style="width:{bar_w}%;background:{bar_col};border-radius:4px;height:8px;transition:width 0.3s;"></div>
    </div>
    <span style="color:{bar_col};font-size:12px;font-weight:600;width:36px;">{util}%</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="color:#8b949e;font-size:12px;width:36px;">VRAM</span>
    <div style="flex:1;background:#21262d;border-radius:4px;height:8px;">
      <div style="width:{int(int(mu)/int(mt)*100)}%;background:#58a6ff;border-radius:4px;height:8px;"></div>
    </div>
    <span style="color:#58a6ff;font-size:12px;font-weight:600;width:80px;">{int(mu):,}/{int(mt):,} MiB</span>
  </div>
</div>""")
        return "\n".join(cards)
    except Exception as e:
        return f'<div style="color:#f85149;padding:12px;">GPU unavailable: {e}</div>'


def get_pipeline_html() -> str:
    steps = [
        ("", "Dataset",   (ROOT/"datasets").exists() and any((ROOT/"datasets").iterdir()) if (ROOT/"datasets").exists() else False),
        ("", "Clean",     bool(list_cleaned_datasets())),
        ("", "Model",     bool(list_models())),
        ("", "Configure", bool(list_configs())),
        ("", "Train",     bool(list_checkpoints())),
        ("", "Eval",      (ROOT/"eval_results").exists() and any(
            (ROOT/"eval_results").iterdir()) if (ROOT/"eval_results").exists() else False),
        ("", "Export",    bool(list_exports())),
    ]
    items = []
    for i, (icon, label, done) in enumerate(steps):
        color  = "#3fb950" if done else "#30363d"
        txt_c  = "#e6edf3" if done else "#8b949e"
        circle = f'<div style="width:32px;height:32px;border-radius:50%;background:{color};display:flex;align-items:center;justify-content:center;font-size:14px;">{"✓" if done else str(i+1)}</div>'
        items.append(f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:4px;">
  {circle}
  <span style="font-size:11px;color:{txt_c};font-weight:{'600' if done else '400'};">{label}</span>
</div>""")
        if i < len(steps) - 1:
            line_col = "#3fb950" if done else "#30363d"
            items.append(f'<div style="flex:1;height:2px;background:{line_col};margin-bottom:16px;"></div>')

    return f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px 24px;margin-bottom:4px;">
  <div style="font-size:12px;color:#8b949e;margin-bottom:14px;font-weight:500;letter-spacing:0.05em;">
    PIPELINE PROGRESS
  </div>
  <div style="display:flex;align-items:center;gap:0;">{''.join(items)}</div>
</div>"""


def get_stats_html() -> str:
    ds  = len(list_datasets())
    cds = len(list_cleaned_datasets())
    mdl = len(list_models())
    cfg = len(list_configs())
    ckp = len(list_checkpoints())
    exp = len(list_exports())

    def card(label, value, sub, color):
        return f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;flex:1;">
  <div style="font-size:28px;font-weight:700;color:{color};">{value}</div>
  <div style="font-size:13px;font-weight:600;color:#e6edf3;margin-top:2px;">{label}</div>
  <div style="font-size:11px;color:#8b949e;margin-top:2px;">{sub}</div>
</div>"""

    return f"""
<div style="display:flex;gap:12px;margin-bottom:4px;">
  {card("Datasets", ds, f"{cds} cleaned", "#58a6ff")}
  {card("Models", mdl, f"{cfg} configs", "#3fb950")}
  {card("Checkpoints", ckp, f"{exp} exports", "#bc8cff")}
</div>"""


def get_guide_html() -> str:
    """Step-by-step guide panel based on what's available on disk."""
    steps = [
        {
            "num": 1, "label": "Add Dataset",
            "done": bool(list_datasets()),
            "tab": "1 · Dataset",
            "action": 'Open <b>1 · Dataset</b> and download a dataset.',
        },
        {
            "num": 2, "label": "Clean Dataset",
            "done": bool(list_cleaned_datasets()),
            "tab": "2 · Clean",
            "action": 'Open <b>2 · Clean</b> and clean your dataset.',
        },
        {
            "num": 3, "label": "Add Model",
            "done": bool(list_models()),
            "tab": "3 · Model",
            "action": 'Open <b>3 · Model</b> and download a base model.',
        },
        {
            "num": 4, "label": "Configure",
            "done": bool(list_configs()),
            "tab": "4 · Configure",
            "action": 'Open <b>4 · Configure</b> and generate a recipe.',
        },
        {
            "num": 5, "label": "Train",
            "done": any(
                any((ROOT/"checkpoints"/d).glob("checkpoint-*"))
                for d in list_checkpoints()
            ) if list_checkpoints() else False,
            "tab": "5 · Train",
            "action": 'Open <b>5 · Train</b> — run Smoke Test, then Start Training.',
        },
        {
            "num": 6, "label": "Evaluate",
            "done": any(
                (ROOT/"eval_results"/d/"summary.md").exists()
                for d in list_checkpoints()
            ) if list_checkpoints() else False,
            "tab": "6 · Eval",
            "action": 'Open <b>6 · Eval</b> and run evaluation.',
        },
        {
            "num": 7, "label": "Export",
            "done": bool(list_exports()) and any(
                any((ROOT/"exports"/d/"merged").glob("model*.safetensors"))
                for d in list_exports()
            ) if list_exports() else False,
            "tab": "7 · Export & Publish",
            "action": 'Open <b>7 · Export &amp; Publish</b> and export your model.',
        },
        {
            "num": 8, "label": "Publish",
            "done": False,
            "tab": "7 · Export & Publish",
            "action": 'Open <b>7 · Export &amp; Publish</b> and publish to HuggingFace.',
        },
    ]

    # Find current step (first not done)
    current = next((s for s in steps if not s["done"]), steps[-1])

    rows = ""
    for s in steps:
        if s["done"]:
            icon  = '<span style="color:#3fb950;font-weight:700;">✓</span>'
            color = "#3fb950"
            text_color = "#8b949e"
            bg = "transparent"
        elif s["num"] == current["num"]:
            icon  = f'<span style="background:#1f6feb;color:#fff;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">{s["num"]}</span>'
            color = "#58a6ff"
            text_color = "#e6edf3"
            bg = "#0d1f3c"
        else:
            icon  = f'<span style="background:#21262d;color:#8b949e;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;">{s["num"]}</span>'
            color = "#484f58"
            text_color = "#484f58"
            bg = "transparent"

        rows += f"""
<div style="display:flex;align-items:center;gap:10px;padding:6px 10px;
            border-radius:6px;background:{bg};margin-bottom:2px;">
  <div style="min-width:20px;">{icon}</div>
  <div style="flex:1;display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;">
    <span style="color:{color};font-size:13px;font-weight:600;">{s["label"]}</span>
    {"<span style='color:#8b949e;font-size:12px;'>" + s["action"] + "</span>" if s["num"] == current["num"] else ""}
  </div>
</div>"""

    return f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;">
  <div style="font-size:12px;color:#8b949e;font-weight:500;letter-spacing:0.05em;margin-bottom:12px;">
    STEP-BY-STEP GUIDE
  </div>
  {rows}
</div>"""


def get_state_html() -> str:
    p = ROOT / "STATE.md"
    if not p.exists():
        return '<div style="color:#8b949e;padding:8px;">STATE.md not found.</div>'
    raw = p.read_text()
    phase  = re.search(r"\*\*Phase:\*\*\s*(.+)", raw)
    status = re.search(r"\*\*Status:\*\*\s*(.+)", raw)
    phase_txt  = phase.group(1).strip()  if phase  else "—"
    status_txt = status.group(1).strip() if status else "—"
    color = "#3fb950" if "✓" in status_txt or "complete" in status_txt.lower() else \
            "#d29922" if "running" in status_txt.lower() else "#58a6ff"
    return f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 18px;">
  <div style="font-size:11px;color:#8b949e;font-weight:500;letter-spacing:0.05em;margin-bottom:8px;">STATUS</div>
  <div style="color:#8b949e;font-size:11px;margin-bottom:2px;">{phase_txt}</div>
  <div style="color:{color};font-size:13px;font-weight:600;">{status_txt}</div>
</div>"""


# ── Next step card ────────────────────────────────────────────────────────────
_NEXT_TAB = {
    "add-dataset":   ("2 · Clean",         "Go to the Clean tab to deduplicate and split your dataset."),
    "clean-dataset": ("3 · Model",          "Go to the Model tab to download a base model."),
    "add-model":     ("4 · Configure",      "Go to Configure to generate your training recipe."),
    "configure":     ("5 · Train",          "Go to Train — run Smoke Test first, then full training."),
    "smoke-test":    ("5 · Train",          "Smoke test passed. Click Start Full Training."),
    "train":         ("6 · Eval",           "Training complete. Go to Eval to benchmark your model."),
    "eval":          ("7 · Export & Publish","Evaluation done. Go to Export & Publish."),
    "export":        ("7 · Export & Publish","Export complete. Edit your Model Card then Publish."),
}

def next_step_card(command: str) -> str:
    tab, msg = _NEXT_TAB.get(command, ("", ""))
    if not tab:
        return ""
    return f"""
<div style="background:#0d2a1a;border:1px solid #238636;border-radius:8px;
            padding:14px 18px;margin-top:12px;">
  <div style="font-size:12px;color:#3fb950;font-weight:600;letter-spacing:0.05em;
              margin-bottom:6px;">COMPLETED — WHAT TO DO NEXT</div>
  <div style="color:#e6edf3;font-size:14px;font-weight:600;margin-bottom:4px;">
    Go to: <span style="color:#58a6ff;">{tab}</span>
  </div>
  <div style="color:#8b949e;font-size:13px;">{msg}</div>
</div>"""


# ── ft command runner ──────────────────────────────────────────────────────────
def run_ft(args: list[str]):
    cmd = [PYTHON, str(SCRIPTS / "ft")] + args
    env = {**os.environ, "FT_PROJECT": str(ROOT),
           "PYTHONUNBUFFERED": "1", "WANDB_DISABLED": "true", "FT_UI": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, cwd=str(ROOT))
    output = ""
    for line in iter(proc.stdout.readline, ""):
        output += line
        yield terminal_wrap(ansi_to_html(output))
    proc.wait()
    rc = proc.returncode
    command = args[0] if args else ""
    if rc == 0:
        tail = '\n<span style="color:#3fb950;font-weight:bold;">✓ Done</span>'
        card = next_step_card(command)
        yield terminal_wrap(ansi_to_html(output) + tail) + card
    else:
        tail = f'\n<span style="color:#f85149;font-weight:bold;">✗ Failed (exit {rc})</span>'
        yield terminal_wrap(ansi_to_html(output) + tail)


def read_sample(name: str) -> str:
    p = ROOT / "datasets" / name / "sample_100.jsonl"
    if not p.exists():
        return terminal_wrap('<span style="color:#d29922;">No sample — run clean-dataset first</span>')
    lines = [json.loads(l) for l in p.read_text().splitlines()[:4] if l.strip()]
    html = ""
    for row in lines:
        for m in row.get("messages", []):
            role_col = {"user": "#58a6ff", "assistant": "#3fb950", "system": "#bc8cff"}.get(m["role"], "#e6edf3")
            html += f'<span style="color:{role_col};font-weight:600;">[{m["role"].upper()}]</span> {m["content"][:300]}<br>'
        html += '<span style="color:#30363d;">──────────────────────────────────</span><br>'
    return terminal_wrap(html, height=320)


def read_data_card(name: str) -> str:
    p = ROOT / "data_cards" / f"{name}.json"
    if not p.exists():
        return terminal_wrap('<span style="color:#d29922;">No data card found</span>')
    try:
        card = json.loads(p.read_text())

        # Only show meaningful fields — skip hashes, internal paths, redundant IDs
        SKIP = {"sha256", "sha256_normalized", "normalized_path", "source_id",
                "rows_after_schema", "rows_after_length", "rows_after_pii"}
        SHOW = [
            ("name",            "Dataset"),
            ("source",          "Source"),
            ("detected_format", "Format"),
            ("rows_raw",        "Raw rows"),
            ("train_rows",      "Train rows"),
            ("val_rows",        "Val rows"),
            ("license",         "License"),
        ]

        rows_html = ""
        for key, label in SHOW:
            val = card.get(key)
            if val is None:
                continue
            # Clean up the "UNKNOWN — fill in before cleaning" noise
            val_str = str(val)
            if "fill in before" in val_str.lower():
                val_str = "UNKNOWN"
            # Format numbers with commas
            if isinstance(val, int):
                val_str = f"{val:,}"
            col = "#3fb950" if key in ("train_rows", "val_rows") else "#e6edf3"
            rows_html += (
                f'<div style="display:flex;justify-content:space-between;'
                f'padding:7px 0;border-bottom:1px solid #21262d;">'
                f'<span style="color:#8b949e;font-size:12px;">{label}</span>'
                f'<span style="color:{col};font-size:12px;font-weight:500;'
                f'max-width:60%;text-align:right;word-break:break-all;">{val_str}</span>'
                f'</div>'
            )

        html = (
            f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
            f'padding:14px 18px;">'
            f'<div style="color:#8b949e;font-size:10px;font-weight:600;letter-spacing:0.06em;'
            f'margin-bottom:10px;">DATA CARD</div>'
            f'{rows_html}'
            f'</div>'
        )
        return html
    except Exception:
        return terminal_wrap(p.read_text())


def read_config_file(name: str) -> str:
    p = ROOT / "configs" / name
    return p.read_text() if p.exists() else "# Config not found"


def read_log(run_name: str) -> str:
    p = ROOT / "logs" / f"train_{run_name}.log"
    if not p.exists():
        return terminal_wrap('<span style="color:#d29922;">No log file yet</span>')

    important = []
    for raw in p.read_text().splitlines():
        line = raw.strip()
        # skip pure tqdm progress bars  (e.g. "  1%|████  | 100/9492 ...")
        if re.match(r"^\s*\d+%\|", line):
            continue
        # skip blank lines and pure separator lines
        if not line or set(line) <= {"-", "=", " "}:
            continue
        # skip verbose axolotl config dump lines (long JSON-like blocks)
        if re.match(r"^\s{2,}\"[a-z_]+\":", line):
            continue
        # keep: metric JSON lines, ERROR/WARNING, ✓/✗ markers, eval lines
        if any(x in line for x in (
            "'loss'", "'eval_loss'", "ERROR", "WARNING", "ABORT",
            "✓", "✗", "Training complete", "Evaluating", "Saving",
            "epoch", "steps_per_second",
        )):
            important.append(line)

    if not important:
        return terminal_wrap('<span style="color:#8b949e;">No log events yet.</span>',
                             height=220)

    # show last 25 important lines
    shown = important[-25:]
    colored = []
    for line in shown:
        if "'loss'" in line and "'eval_loss'" not in line:
            try:
                d = ast.literal_eval(line)
                step_n = len([l for l in important if "'loss'" in l and "'eval_loss'" not in l])
                loss_v = float(d.get("loss", 0))
                ppl_v  = float(d.get("ppl", 0))
                epoch_v= float(d.get("epoch", 0))
                lc = "32" if loss_v < 1.5 else ("33" if loss_v < 3.0 else "31")
                colored.append(
                    f'\033[2mstep {step_n:>4}\033[0m  '
                    f'loss \033[{lc}m{loss_v:.4f}\033[0m  '
                    f'ppl \033[36m{ppl_v:.3f}\033[0m  '
                    f'epoch \033[2m{epoch_v:.3f}\033[0m'
                )
                continue
            except Exception:
                pass
        if "'eval_loss'" in line:
            try:
                d = ast.literal_eval(line)
                el = float(d.get("eval_loss", 0))
                ep = float(d.get("eval_ppl", 0))
                colored.append(
                    f'\033[35m[EVAL]\033[0m  '
                    f'eval_loss \033[32m{el:.4f}\033[0m  '
                    f'eval_ppl \033[36m{ep:.3f}\033[0m'
                )
                continue
            except Exception:
                pass
        colored.append(line)

    return terminal_wrap(ansi_to_html("\n".join(colored)), height=260)


def get_loss_df(run_name: str) -> pd.DataFrame:
    """Parse training log and return DataFrame for loss chart."""
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
    """Parse training log → compact live metrics panel with ETA."""
    if not run_name:
        return ""
    p = ROOT / "logs" / f"train_{run_name}.log"
    if not p.exists():
        return ""

    text  = p.read_text()
    lines = text.splitlines()

    # ── Parse latest training-step JSON line ────────────────────────────────
    loss = grad = lr = epoch = ppl = None
    step = 0
    for line in lines:
        if "'loss'" in line and "'eval_loss'" not in line:
            step += 1
            try:
                d = ast.literal_eval(line.strip())
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

    # ── ETA from tqdm lines: "100/9492 [00:10<24:57:50" ────────────────────
    eta_str = elapsed_str = "—"
    total_steps = step
    tqdm_m = re.search(
        r"(\d+)/(\d+)\s*\[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?)", text
    )
    if tqdm_m:
        cur, total_steps = int(tqdm_m.group(1)), int(tqdm_m.group(2))

        def _secs(s: str) -> int:
            parts = list(map(int, s.split(":")))
            return parts[0]*3600 + parts[1]*60 + (parts[2] if len(parts)==3 else 0)

        elapsed_s  = _secs(tqdm_m.group(3))
        remaining_s = _secs(tqdm_m.group(4))
        eh, em = elapsed_s // 3600, (elapsed_s % 3600) // 60
        rh, rm = remaining_s // 3600, (remaining_s % 3600) // 60
        elapsed_str = f"{eh}h {em}m" if eh else f"{em}m"
        eta_str     = f"~{rh}h {rm}m" if rh else (f"~{rm}m" if rm else "< 1m")

    pct = round(step / total_steps * 100) if total_steps else 0

    # ── Color helpers ────────────────────────────────────────────────────────
    lc = "#3fb950" if (loss or 9) < 1.5 else ("#d29922" if (loss or 9) < 3.0 else "#f85149")
    gc = "#3fb950" if (grad or 9) < 2.0 else ("#d29922" if (grad or 9) < 5.0 else "#f85149")

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

    bar_w = f"{pct}%"
    bar_c = "#3fb950" if pct > 80 else ("#58a6ff" if pct > 20 else "#d29922")

    return f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:18px 20px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
    <span style="color:#8b949e;font-size:11px;font-weight:600;letter-spacing:0.06em;">
      LIVE METRICS
    </span>
    <span style="color:#8b949e;font-size:11px;">{run_name}</span>
  </div>

  <!-- progress bar -->
  <div style="background:#21262d;border-radius:4px;height:6px;margin-bottom:16px;">
    <div style="background:{bar_c};width:{bar_w};height:6px;border-radius:4px;
                transition:width 0.3s;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-bottom:16px;
              font-size:11px;color:#8b949e;">
    <span>Step <b style="color:#c9d1d9;">{step}</b> / {total_steps}</span>
    <span>{pct}% complete</span>
    <span>Elapsed <b style="color:#c9d1d9;">{elapsed_str}</b></span>
    <span>ETA <b style="color:#d29922;">{eta_str}</b></span>
  </div>

  <!-- metric cards -->
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    {card("Loss",      f"{loss:.4f}" if loss else "—",  lc)}
    {card("Perplexity",f"{ppl:.3f}"  if ppl  else "—",  "#39c5cf")}
    {card("Grad Norm", f"{grad:.3f}" if grad else "—",  gc)}
    {card("Epoch",     f"{epoch:.3f}" if epoch else "—", "#e6edf3")}
    {card("Learn Rate",f"{lr:.2e}"   if lr   else "—",  "#bc8cff")}
  </div>
</div>"""


def read_eval_summary(run_name: str) -> str:
    p = ROOT / "eval_results" / run_name / "summary.md"
    if not p.exists():
        return terminal_wrap('<span style="color:#d29922;">No eval results yet. Run evaluation first.</span>')
    return terminal_wrap(ansi_to_html(p.read_text()), height=360)


def read_model_card(name: str) -> str:
    p = ROOT / "exports" / name / "MODEL_CARD.md"
    return p.read_text() if p.exists() else "# MODEL_CARD.md not found\n\nRun export first."


def save_model_card(name: str, content: str) -> str:
    p = ROOT / "exports" / name / "MODEL_CARD.md"
    if not p.parent.exists():
        return "Export directory not found."
    p.write_text(content)
    return "✓ Saved"


# ── Evaluation helpers ────────────────────────────────────────────────────────

def list_eval_runs() -> list[str]:
    eval_dir = ROOT / "eval_results"
    if not eval_dir.exists():
        return []
    return sorted(
        [p.name for p in eval_dir.iterdir() if p.is_dir()],
        reverse=True,
    )


def parse_training_log(run_name: str) -> dict:
    """Return lists of (step, loss, ppl) for train and eval points."""
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
        if "loss" in d and "eval_loss" not in d:
            step += 1
            train_steps.append(step)
            train_loss.append(float(d["loss"]))
            train_ppl.append(float(d.get("ppl", 0)) or float(d["loss"]) ** 2.718)
        elif "eval_loss" in d:
            eval_steps.append(step)
            eval_loss.append(float(d["eval_loss"]))
            eval_ppl.append(float(d.get("eval_ppl", 0)) or float(d["eval_loss"]) ** 2.718)
    return dict(
        train_steps=train_steps, train_loss=train_loss, train_ppl=train_ppl,
        eval_steps=eval_steps,   eval_loss=eval_loss,   eval_ppl=eval_ppl,
    )


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
_CHART_W   = 4.8   # identical width for all 3 charts
_CHART_H   = 3.6   # identical height for all 3 charts
_DPI       = 100   # 480×360px — fits cleanly in a 3-column Gradio row


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
    """Put value labels at first and last point."""
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
    if data.get("train_loss"):
        bl = data["train_loss"][0];  al = data["train_loss"][-1]
        bp = data["train_ppl"][0];   ap = data["train_ppl"][-1]
        if data.get("eval_loss") and len(data["eval_loss"]) >= 2:
            al = data["eval_loss"][-1]
        if data.get("eval_ppl") and len(data["eval_ppl"]) >= 2:
            ap = data["eval_ppl"][-1]
        x, w = [0, 1], 0.30
        b1 = ax.bar([i - w/2 for i in x], [bl, bp], w,
                    color=_GREY_BAR, label="Before", zorder=3,
                    edgecolor=_GRID_COL, linewidth=0.5)
        b2 = ax.bar([i + w/2 for i in x], [al, ap], w,
                    color=_PURPLE,   label="After",  zorder=3,
                    edgecolor=_GRID_COL, linewidth=0.5)
        max_y = max(bl, bp)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + max_y * 0.015, f"{h:.2f}",
                    ha="center", va="bottom",
                    color=_TITLE_COL, fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Loss", "Perplexity"],
                           color=_TICK_COL, fontsize=9.5)
        ax.set_ylim(0, max_y * 1.28)
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
    rows = [
        ("Train Loss",       f"{bl:.4f}", f"{al:.4f}", f"{(al-bl)/bl*100:+.1f}%"),
        ("Train Perplexity", f"{bp:.3f}", f"{ap:.3f}", f"{(ap-bp)/bp*100:+.1f}%"),
    ]
    if data.get("eval_loss") and len(data["eval_loss"]) >= 2:
        el_b = data["eval_loss"][0]; el_a = data["eval_loss"][-1]
        rows.append(("Eval Loss", f"{el_b:.4f}", f"{el_a:.4f}",
                     f"{(el_a-el_b)/el_b*100:+.1f}%"))
    if data.get("eval_ppl") and len(data["eval_ppl"]) >= 2:
        ep_b = data["eval_ppl"][0]; ep_a = data["eval_ppl"][-1]
        rows.append(("Eval Perplexity", f"{ep_b:.3f}", f"{ep_a:.3f}",
                     f"{(ep_a-ep_b)/ep_b*100:+.1f}%"))
    return pd.DataFrame(rows, columns=cols)


def load_eval_run(run_name: str):
    empty = pd.DataFrame(columns=["Metric", "Before fine-tuning",
                                   "After fine-tuning", "Change"])
    if not run_name:
        return build_loss_chart({}), build_ppl_chart({}), build_summary_chart({}), empty
    data = parse_training_log(run_name)
    return (build_loss_chart(data), build_ppl_chart(data),
            build_summary_chart(data), build_metrics_df(data))


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
footer { display:none !important; }
* { box-sizing: border-box; }

/* ── Typography ───────────────────────────────────────────────────────────── */
body, .gradio-container, .gradio-container * {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial,
               sans-serif !important;
  -webkit-font-smoothing: antialiased !important;
}

/* ── Inputs & Textareas ───────────────────────────────────────────────────── */
input:not([type=range]):not([type=checkbox]):not([type=radio]),
textarea, select {
  color: #e6edf3 !important;
  background-color: #161b22 !important;
  border: 1px solid #30363d !important;
  border-radius: 6px !important;
  caret-color: #e6edf3 !important;
  font-size: 13px !important;
}
input:not([type=range])::placeholder, textarea::placeholder {
  color: #6e7681 !important;
  opacity: 1 !important;
}
input:not([type=range]):focus, textarea:focus {
  border-color: #58a6ff !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(88,166,255,0.12) !important;
}
/* Gradio svelte wrappers */
[class*="svelte"] input:not([type=range]):not([type=checkbox]):not([type=radio]),
[class*="svelte"] textarea,
.block input:not([type=range]):not([type=checkbox]):not([type=radio]),
.block textarea, .form input:not([type=range]):not([type=checkbox]):not([type=radio]),
.form textarea {
  color: #e6edf3 !important;
  background-color: #161b22 !important;
}

/* ── Labels ───────────────────────────────────────────────────────────────── */
.block label span, .block > label, .form > label,
span.text-sm, label > span, .label-wrap span,
.block label > span:first-child {
  color: #c9d1d9 !important;
  font-size: 13px !important;
  font-weight: 500 !important;
}
.block .info, .block span.text-xs { color: #8b949e !important; font-size: 11.5px !important; }

/* ── Buttons ──────────────────────────────────────────────────────────────── */
button { border-radius: 6px !important; font-weight: 500 !important;
         transition: background 0.15s, border-color 0.15s !important; }
button.primary, button[data-testid="primary"] {
  background: #238636 !important; border: 1px solid #238636 !important;
  color: #ffffff !important; font-weight: 600 !important;
}
button.primary:hover { background: #2ea043 !important; border-color: #2ea043 !important; }
button.secondary, button[data-testid="secondary"] {
  background: #21262d !important; border: 1px solid #30363d !important;
  color: #c9d1d9 !important;
}
button.secondary:hover { background: #30363d !important; }
button.stop, button[data-testid="stop"] {
  background: #3d1a1a !important; border: 1px solid #f85149 !important;
  color: #f85149 !important;
}
#land_dev_btn {
  background: #0d1b2a !important; border: 1px solid #58a6ff !important;
  color: #58a6ff !important; font-weight: 600 !important;
}
#land_dev_btn:hover {
  background: #1a2d42 !important; border-color: #79c0ff !important;
  color: #79c0ff !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.tab-nav { border-bottom: 1px solid #30363d !important; }
.tab-nav button {
  color: #8b949e !important; background: transparent !important;
  border-bottom: 2px solid transparent !important;
  font-size: 13px !important; padding: 8px 16px !important;
}
.tab-nav button.selected {
  color: #e6edf3 !important; border-bottom-color: #58a6ff !important;
  background: transparent !important; font-weight: 600 !important;
}
.tab-nav button:hover { color: #c9d1d9 !important; }

/* ── Dropdowns ────────────────────────────────────────────────────────────── */
ul[role="listbox"], ul.options, .dropdown-menu,
[data-testid="dropdown-options"], li[role="option"] {
  background: #1c2128 !important; color: #e6edf3 !important;
  border: 1px solid #30363d !important;
}
li[role="option"] {
  padding: 8px 14px !important; font-size: 13px !important;
}
li[role="option"]:hover, li[role="option"].selected {
  background: #2d333b !important; color: #e6edf3 !important;
}
li[role="option"] * { color: #e6edf3 !important; }
select option { background: #1c2128 !important; color: #e6edf3 !important; }

/* ── Popovers / Overflow menus ────────────────────────────────────────────── */
[popover], [popover] *, [role="menu"], [role="menu"] *,
[class*="overflow"], div[style*="position: absolute"],
div[style*="position:absolute"] {
  background-color: #1c2128 !important;
  color: #e6edf3 !important; border-color: #30363d !important;
}
button[aria-haspopup] { color: #8b949e !important; }

/* ── DataFrame / Table ────────────────────────────────────────────────────── */
.dataframe, .table-wrap, [data-testid="dataframe"] {
  background: #0d1117 !important;
  border: 1px solid #30363d !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}
table { border-collapse: collapse !important; width: 100% !important; }
thead tr { background: #1c2128 !important; }
thead tr th {
  color: #c9d1d9 !important;
  font-size: 11.5px !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  padding: 11px 16px !important;
  border-bottom: 2px solid #30363d !important;
  background: #1c2128 !important;
  white-space: nowrap !important;
}
tbody tr td {
  color: #e6edf3 !important;
  font-size: 13px !important;
  padding: 10px 16px !important;
  border-bottom: 1px solid #21262d !important;
  background: transparent !important;
}
tbody tr:nth-child(even) td { background: rgba(22,27,34,0.6) !important; }
tbody tr:hover td { background: #21262d !important; }

/* ── Checkboxes & Radios ──────────────────────────────────────────────────── */
input[type=checkbox], input[type=radio] {
  accent-color: #238636 !important;
  width: 15px !important; height: 15px !important;
}
.block label:has(input[type=checkbox]) span,
.block label:has(input[type=radio]) span {
  color: #c9d1d9 !important;
}

/* ── Sliders ──────────────────────────────────────────────────────────────── */
input[type=range] { accent-color: #58a6ff !important; }
input[type=range]::-webkit-slider-thumb {
  background: #58a6ff !important; width: 16px !important; height: 16px !important;
}
input[type=range]::-webkit-slider-runnable-track {
  background: #30363d !important; height: 4px !important; border-radius: 2px !important;
}

/* ── Number & Password inputs (explicit overrides) ────────────────────────── */
input[type=number], input[type=password] {
  color: #e6edf3 !important;
  background-color: #161b22 !important;
  border: 1px solid #30363d !important;
  -webkit-text-fill-color: #e6edf3 !important;
}
input[type=password]::-webkit-credentials-auto-fill-button { display: none !important; }
/* Catch Gradio theme overriding password on focus/autofill */
input[type=password]:-webkit-autofill,
input[type=password]:-webkit-autofill:focus {
  -webkit-box-shadow: 0 0 0 1000px #161b22 inset !important;
  -webkit-text-fill-color: #e6edf3 !important;
}

/* ── Plot / Chart containers ─────────────────────────────────────────────── */
.gradio-plot, .plot-container, [data-testid="plot"],
.block:has(.gradio-plot), .block:has(canvas) {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.gradio-plot img, .plot-container img, [data-testid="plot"] img,
.gradio-plot canvas, .plot-container canvas {
  max-width: 100% !important;
  width: 100% !important;
  height: auto !important;
  display: block !important;
  border-radius: 8px !important;
}

/* ── Blocks ───────────────────────────────────────────────────────────────── */
.block { border-color: #30363d !important; border-radius: 8px !important; }

/* ── Scrollbars ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }

/* ── Code ─────────────────────────────────────────────────────────────────── */
.cm-content, .cm-line { color: #e6edf3 !important; }
pre, code {
  background: #161b22 !important; color: #e6edf3 !important;
  border-radius: 4px !important; border: 1px solid #30363d !important;
}

/* ── Chat bubbles ─────────────────────────────────────────────────────────── */
.message-bubble-border { border-color: #30363d !important; }
.bot, .bot * { background: #161b22 !important; color: #e6edf3 !important; }
.user, .user * { background: #1f6feb !important; color: #ffffff !important; }
[data-testid="bot"] { background: #161b22 !important; color: #e6edf3 !important; }
[data-testid="user"] { background: #1f6feb !important; color: #ffffff !important; }
.message { color: #e6edf3 !important; }
.message p, .message span, .message div,
.prose p, .prose span, .prose li { color: #e6edf3 !important; }

/* ── Accordion ────────────────────────────────────────────────────────────── */
.accordion-header { color: #c9d1d9 !important; background: #161b22 !important;
                    border-color: #30363d !important; }

/* ── File upload ──────────────────────────────────────────────────────────── */
.upload-container, [data-testid="upload"] {
  border: 2px dashed #30363d !important; background: #161b22 !important;
  border-radius: 8px !important;
}
[data-testid="upload"]:hover { border-color: #58a6ff !important; }
"""

# ── Developer credentials (change these) ─────────────────────────────────────
DEV_USERNAME = os.environ.get("DEV_USERNAME", "mkd-dev")
DEV_PASSWORD = os.environ.get("DEV_PASSWORD", "mkd2026!")

# ── App ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MKD Fine-Tuning Platform") as app:

    # ── Landing page ──────────────────────────────────────────────────────────
    with gr.Group(visible=True) as landing_group:
        gr.HTML("""
        <div style="padding:60px 20px;text-align:center;">
          <div style="width:56px;height:56px;background:#1f6feb;border-radius:12px;
                      display:inline-flex;align-items:center;justify-content:center;
                      font-size:22px;font-weight:700;color:#fff;margin-bottom:20px;">MKD</div>
          <h1 style="margin:0 0 8px;font-size:32px;font-weight:700;color:#e6edf3;">
            MKD Fine-Tuning Platform
          </h1>
          <p style="color:#8b949e;font-size:16px;margin:0 0 48px;">
            Fine-tune any LLM on your data — who are you?
          </p>
          <!-- Two main cards — equal height via align-items:stretch -->
          <div style="display:flex;gap:24px;justify-content:center;max-width:760px;
                      margin:0 auto;align-items:stretch;">

            <div style="flex:1;min-width:280px;background:#161b22;border:1px solid #30363d;
                        border-radius:12px;padding:32px;text-align:left;
                        display:flex;flex-direction:column;">
              <div style="font-size:24px;margin-bottom:10px;">&#9881;</div>
              <h2 style="margin:0 0 8px;color:#e6edf3;font-size:20px;">Developer</h2>
              <p style="color:#8b949e;font-size:14px;line-height:1.6;margin:0 0 20px;flex:1;">
                Full control over training — custom configs, live metrics,
                eval benchmarks, and advanced export.
              </p>
              <div style="padding:6px 12px;background:#21262d;border-radius:6px;
                          display:inline-block;color:#8b949e;font-size:12px;">
                For engineers and ML practitioners
              </div>
            </div>

            <div style="flex:1;min-width:280px;background:#0d2a1a;border:1px solid #238636;
                        border-radius:12px;padding:32px;text-align:left;
                        display:flex;flex-direction:column;">
              <div style="font-size:24px;margin-bottom:10px;">&#9654;</div>
              <h2 style="margin:0 0 8px;color:#e6edf3;font-size:20px;">General User</h2>
              <p style="color:#8b949e;font-size:14px;line-height:1.6;margin:0 0 20px;flex:1;">
                Simple 3-step flow — pick a preset, upload data, click Start.
                All settings pre-configured. No coding needed.
              </p>
              <div style="padding:6px 12px;background:#0d2a1a;border:1px solid #238636;
                          border-radius:6px;display:inline-block;color:#3fb950;font-size:12px;">
                No ML knowledge required
              </div>
            </div>

          </div>

          <!-- Evaluation — secondary link below the main cards -->
          <div style="max-width:760px;margin:20px auto 0;text-align:center;">
            <div style="display:inline-flex;align-items:center;gap:10px;
                        background:#0d1b2a;border:1px solid #1f6feb;border-radius:8px;
                        padding:12px 24px;">
              <span style="font-size:16px;">&#128202;</span>
              <span style="color:#8b949e;font-size:14px;">Already trained a model?</span>
              <span style="color:#58a6ff;font-size:14px;font-weight:600;">View Evaluation →</span>
            </div>
          </div>
        </div>
        """)
        with gr.Row():
            land_dev_btn  = gr.Button("Enter as Developer",    variant="secondary", scale=1, elem_id="land_dev_btn")
            land_user_btn = gr.Button("Enter as General User", variant="primary",   scale=1)
            land_eval_btn = gr.Button("View Evaluation",       variant="secondary", scale=1)

    # ── Developer login screen ────────────────────────────────────────────────
    with gr.Group(visible=False) as dev_login_group:
        gr.HTML("""
        <div style="max-width:380px;margin:60px auto;padding:0 20px;text-align:center;">
          <div style="width:48px;height:48px;background:#1f6feb;border-radius:10px;
                      display:inline-flex;align-items:center;justify-content:center;
                      font-size:18px;font-weight:700;color:#fff;margin-bottom:20px;">MKD</div>
          <h2 style="color:#e6edf3;margin:0 0 6px;font-size:22px;">Developer Access</h2>
          <p style="color:#8b949e;font-size:14px;margin:0 0 28px;">
            Enter your credentials to continue
          </p>
        </div>
        """)
        with gr.Column(elem_id="login-box"):
            dev_login_user = gr.Textbox(label="Username", placeholder="username")
            dev_login_pass = gr.Textbox(label="Password", placeholder="password",
                                        type="password")
            dev_login_err  = gr.HTML("")
            with gr.Row():
                dev_login_back_btn = gr.Button("← Back",  variant="secondary", scale=1)
                dev_login_btn      = gr.Button("Login",    variant="primary",   scale=2)

    # ── Developer mode ────────────────────────────────────────────────────────
    dev_content_group = gr.Group(visible=False)
    with gr.Group(visible=False) as dev_group:
        with gr.Row():
            gr.HTML('<span style="color:#8b949e;font-size:12px;padding:4px 0;">Developer Mode</span>')
            back_dev_btn = gr.Button("← Home", size="sm", variant="secondary", scale=0)

    # ── General user mode ─────────────────────────────────────────────────────
    with gr.Group(visible=False) as user_group:
        with gr.Row():
            gr.HTML('<span style="color:#8b949e;font-size:12px;padding:4px 0;">General User Mode</span>')
            back_user_btn   = gr.Button("← Home",            size="sm", variant="secondary",  scale=0, visible=True)
            cancel_train_btn = gr.Button("Cancel Training",  size="sm", variant="stop",        scale=0, visible=False)

        # Screen 1: Choose preset
        with gr.Group(visible=True) as user_screen1:
            gr.HTML("""
            <div style="padding:32px 0 24px;text-align:center;">
              <h2 style="color:#e6edf3;margin:0 0 8px;font-size:24px;font-weight:700;">
                How would you like to start?
              </h2>
              <p style="color:#8b949e;font-size:14px;margin:0;">
                Choose a path below. Both will guide you step by step.
              </p>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="background:#0d2a1a;border:2px solid #238636;border-radius:12px;
                                padding:28px 28px 20px;position:relative;margin-bottom:8px;min-height:160px;">
                      <div style="position:absolute;top:14px;right:14px;background:#238636;
                                  color:#fff;font-size:10px;font-weight:700;padding:3px 10px;
                                  border-radius:20px;letter-spacing:0.06em;">RECOMMENDED</div>
                      <div style="color:#e6edf3;font-size:18px;font-weight:700;margin-bottom:10px;">Quick Start</div>
                      <div style="color:#8b949e;font-size:14px;line-height:1.7;margin-bottom:14px;">
                        Upload your data and give it a name.<br>
                        We handle everything else automatically.
                      </div>
                      <div style="color:#3fb950;font-size:12px;font-weight:500;">
                        Test run: ~1 min &nbsp;&nbsp;|&nbsp;&nbsp; Full run: 3–6 h
                      </div>
                    </div>
                    """)
                    user_qs_btn = gr.Button("Start with Quick Start", variant="primary")

                with gr.Column(scale=1):
                    gr.HTML("""
                    <div style="background:#161b22;border:2px solid #30363d;border-radius:12px;
                                padding:28px 28px 20px;margin-bottom:8px;min-height:160px;">
                      <div style="color:#e6edf3;font-size:18px;font-weight:700;margin-bottom:10px;">Choose model &amp; data</div>
                      <div style="color:#8b949e;font-size:14px;line-height:1.7;margin-bottom:14px;">
                        Pick which AI model and which dataset<br>
                        to use from what is already on the server.
                      </div>
                      <div style="color:#8b949e;font-size:12px;font-weight:500;">
                        All downloaded models &nbsp;&nbsp;|&nbsp;&nbsp; All datasets
                      </div>
                    </div>
                    """)
                    user_cm_btn = gr.Button("Choose model & data", variant="secondary")

        # Screen 2a: Quick Start config
        with gr.Group(visible=False) as user_screen2a:
            gr.HTML("""
            <div style="padding:8px 0 4px;">
              <div style="display:flex;align-items:center;gap:0;margin-bottom:24px;">
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#238636;color:#fff;font-size:13px;
                              font-weight:700;">1</div>
                  <div style="color:#3fb950;font-size:12px;margin-top:4px;font-weight:600;">Setup</div>
                </div>
                <div style="flex:2;height:2px;background:#30363d;margin-bottom:16px;"></div>
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#21262d;border:2px solid #30363d;
                              color:#8b949e;font-size:13px;font-weight:700;">2</div>
                  <div style="color:#8b949e;font-size:12px;margin-top:4px;">Training</div>
                </div>
                <div style="flex:2;height:2px;background:#30363d;margin-bottom:16px;"></div>
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#21262d;border:2px solid #30363d;
                              color:#8b949e;font-size:13px;font-weight:700;">3</div>
                  <div style="color:#8b949e;font-size:12px;margin-top:4px;">Chat</div>
                </div>
              </div>
              <h2 style="color:#e6edf3;margin:0 0 6px;font-size:22px;font-weight:700;">
                Set up your model
              </h2>
              <p style="color:#8b949e;font-size:14px;margin:0 0 24px;line-height:1.6;">
                Give your model a name, choose your training data, then press Start.
              </p>
            </div>
            """)

            with gr.Column(elem_id="qs-center"):
                user_qs_name = gr.Textbox(
                    label="Model name",
                    placeholder="e.g.  my-assistant  or  customer-support-bot",
                    max_lines=1,
                    info="A short, lowercase name with no spaces. Used to identify your model.",
                )

                gr.HTML('<div style="height:4px;"></div>')

                user_qs_dataset = gr.Radio(
                    ["Use the built-in example dataset",
                     "Upload my own data file"],
                    value="Use the built-in example dataset",
                    label="Training data",
                    info="The built-in dataset is good for a quick test. Upload your own data for real results.",
                )
                user_qs_file = gr.File(
                    label="Your data file  (.jsonl, .json or .csv)",
                    file_types=[".jsonl", ".json", ".csv"],
                    visible=False,
                )

                gr.HTML('<div style="height:4px;"></div>')

                user_qs_test_mode = gr.Checkbox(
                    label="Do a quick test run first",
                    value=True,
                    info="Recommended — confirms everything works before the full training run.",
                )
                user_qs_test_steps = gr.Number(
                    label="Quick test steps",
                    value=50,
                    precision=0,
                    minimum=10,
                    maximum=500,
                    info="Number of steps for the quick test. More steps = better evaluation metrics.",
                    visible=True,
                )

                gr.HTML('<div style="height:20px;"></div>')

                user_qs_start_btn = gr.Button(
                    "Start Training", variant="primary", size="lg")

                gr.HTML('<div style="height:12px;"></div>')
                user_qs_back_btn = gr.Button("← Back", variant="secondary", size="sm")
                user_qs_out = gr.HTML("", visible=False)

        # Screen 2b: Custom Mix config — fully dynamic from server
        def get_model_choices():
            choices = []
            for m in list_models():
                info_f = ROOT / "models" / m / "MODEL_INFO.json"
                try:
                    info = json.loads(info_f.read_text())
                    hf_id  = info.get("model_id", m)
                    params = f'{info.get("params_billion", "?")}B'
                    label  = hf_id.split("/")[-1].replace("-Instruct","").replace("-instruct","")
                    choices.append(f"{label}  ({params})")
                except Exception:
                    choices.append(m)
            return choices or ["Phi-3.5-mini-instruct  (3.8B)"]

        def get_cm_dataset_choices():
            choices = []
            for d in list_cleaned_datasets():
                card_f = ROOT / "data_cards" / f"{d}.json"
                try:
                    card = json.loads(card_f.read_text())
                    rows = card.get("train_rows", "?")
                    choices.append(f"{d}  ({rows:,} examples)" if isinstance(rows, int) else d)
                except Exception:
                    choices.append(d)
            choices.append("Upload my own data")
            return choices

        with gr.Group(visible=False) as user_screen2b:
            gr.HTML("""
            <div style="padding:8px 0 4px;">
              <div style="display:flex;align-items:center;gap:0;margin-bottom:24px;">
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#238636;color:#fff;font-size:13px;
                              font-weight:700;">1</div>
                  <div style="color:#3fb950;font-size:12px;margin-top:4px;font-weight:600;">Setup</div>
                </div>
                <div style="flex:2;height:2px;background:#30363d;margin-bottom:16px;"></div>
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#21262d;border:2px solid #30363d;
                              color:#8b949e;font-size:13px;font-weight:700;">2</div>
                  <div style="color:#8b949e;font-size:12px;margin-top:4px;">Training</div>
                </div>
                <div style="flex:2;height:2px;background:#30363d;margin-bottom:16px;"></div>
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#21262d;border:2px solid #30363d;
                              color:#8b949e;font-size:13px;font-weight:700;">3</div>
                  <div style="color:#8b949e;font-size:12px;margin-top:4px;">Chat</div>
                </div>
              </div>
              <h2 style="color:#e6edf3;margin:0 0 6px;font-size:22px;font-weight:700;">Choose model &amp; data</h2>
              <p style="color:#8b949e;font-size:14px;margin:0 0 20px;">
                Select a model and dataset from the server. Everything else is handled automatically.
              </p>
            </div>
            """)
            with gr.Row():
                user_cm_model = gr.Radio(
                    choices=get_model_choices(),
                    value=get_model_choices()[0],
                    label="Which AI model do you want to fine-tune?",
                    scale=3,
                )
                cm_model_refresh = gr.Button("↺", size="sm", variant="secondary", scale=0)

            gr.HTML('<div style="height:8px;"></div>')

            with gr.Row():
                user_cm_dataset = gr.CheckboxGroup(
                    choices=get_cm_dataset_choices(),
                    label="What data do you want to train on?  (select one or more)",
                    scale=3,
                )
                cm_ds_refresh = gr.Button("↺", size="sm", variant="secondary", scale=0)

            user_cm_file = gr.File(
                label="Upload your data file  (.jsonl, .json or .csv)",
                file_types=[".jsonl", ".json", ".csv"],
                visible=False,
            )

            gr.HTML('<div style="height:8px;"></div>')
            user_cm_name = gr.Textbox(
                label="What should we call your model?",
                placeholder="e.g.  my-assistant  or  support-bot")

            gr.HTML('<div style="height:8px;"></div>')
            with gr.Row():
                user_cm_epochs = gr.Number(
                    label="Epochs",
                    value=1,
                    precision=0,
                    minimum=1,
                    maximum=10,
                    info="Full training epochs. 1 = fast (~1-3 hrs), 3 = thorough (~9 hrs).",
                    scale=1,
                )
                user_cm_test_steps = gr.Number(
                    label="Quick test steps",
                    value=50,
                    precision=0,
                    minimum=10,
                    maximum=500,
                    info="Steps for the quick check before full training.",
                    scale=1,
                )

            gr.HTML('<div style="height:16px;"></div>')
            with gr.Row():
                user_cm_back_btn  = gr.Button("← Back",        variant="secondary", scale=1)
                user_cm_start_btn = gr.Button("Start Training", variant="primary",   scale=2)

            user_cm_out = gr.HTML("", visible=False)

            cm_model_refresh.click(lambda: gr.Radio(choices=get_model_choices()),       outputs=user_cm_model)
            cm_ds_refresh.click(lambda: gr.CheckboxGroup(choices=get_cm_dataset_choices()), outputs=user_cm_dataset)

        # Screen 3: Progress
        with gr.Group(visible=False) as user_screen3:
            gr.HTML("""
            <div style="padding:8px 0 4px;">
              <div style="display:flex;align-items:center;gap:0;margin-bottom:24px;">
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#238636;color:#fff;font-size:13px;
                              font-weight:700;">&#10003;</div>
                  <div style="color:#3fb950;font-size:12px;margin-top:4px;font-weight:600;">Setup</div>
                </div>
                <div style="flex:2;height:2px;background:#238636;margin-bottom:16px;"></div>
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#1f6feb;color:#fff;font-size:13px;
                              font-weight:700;">2</div>
                  <div style="color:#58a6ff;font-size:12px;margin-top:4px;font-weight:600;">Training</div>
                </div>
                <div style="flex:2;height:2px;background:#30363d;margin-bottom:16px;"></div>
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#21262d;border:2px solid #30363d;
                              color:#8b949e;font-size:13px;font-weight:700;">3</div>
                  <div style="color:#8b949e;font-size:12px;margin-top:4px;">Chat</div>
                </div>
              </div>
            </div>
            """)
            user_progress_html = gr.HTML("")
            user_progress_out  = gr.HTML("", visible=False)

            gr.HTML('<div style="margin-top:16px;color:#8b949e;font-size:12px;font-weight:600;'
                    'letter-spacing:.05em;text-transform:uppercase;">GPU Status</div>')
            user_gpu_html = gr.HTML(get_gpu_html())
            user_gpu_timer = gr.Timer(value=5)
            user_gpu_timer.tick(get_gpu_html, outputs=user_gpu_html)

        # Screen 4: Inline chat
        with gr.Group(visible=False) as user_screen4:
            gr.HTML("""
            <div style="padding:8px 0 4px;">
              <div style="display:flex;align-items:center;gap:0;margin-bottom:20px;">
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#238636;color:#fff;font-size:13px;
                              font-weight:700;">&#10003;</div>
                  <div style="color:#3fb950;font-size:12px;margin-top:4px;font-weight:600;">Setup</div>
                </div>
                <div style="flex:2;height:2px;background:#238636;margin-bottom:16px;"></div>
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#238636;color:#fff;font-size:13px;
                              font-weight:700;">&#10003;</div>
                  <div style="color:#3fb950;font-size:12px;margin-top:4px;font-weight:600;">Training</div>
                </div>
                <div style="flex:2;height:2px;background:#238636;margin-bottom:16px;"></div>
                <div style="flex:1;text-align:center;">
                  <div style="display:inline-block;width:28px;height:28px;line-height:28px;
                              border-radius:50%;background:#1f6feb;color:#fff;font-size:13px;
                              font-weight:700;">3</div>
                  <div style="color:#58a6ff;font-size:12px;margin-top:4px;font-weight:600;">Chat</div>
                </div>
              </div>
            </div>
            <div style="background:#0d2a1a;border:1px solid #238636;border-radius:10px;
                        padding:16px 20px;margin-bottom:16px;">
              <div style="font-size:17px;font-weight:700;color:#3fb950;margin-bottom:4px;">
                Your model is ready
              </div>
              <div style="color:#8b949e;font-size:13px;">
                Type a message below to test how your fine-tuned model responds.
              </div>
            </div>
            """)
            user_chatbot = gr.Chatbot(height=380, label="Chat with your fine-tuned model")
            with gr.Row():
                user_chat_input = gr.Textbox(
                    placeholder="Type your message and press Enter...",
                    label="", scale=4, lines=1,
                )
                user_send_btn = gr.Button("Send", variant="primary", scale=0)
            gr.HTML('<p style="color:#484f58;font-size:11px;margin:4px 0 12px;">'
                    'Press Enter or click Send · Clear button resets the conversation</p>')
            with gr.Row():
                user_clear_btn  = gr.Button("Clear conversation", size="sm", variant="secondary")
                user_new_btn    = gr.Button("Train a new model", size="sm", variant="secondary")
                user_publish_btn = gr.Button("Publish to HuggingFace", size="sm", variant="primary")
            user_publish_repo = gr.Textbox(
                label="HuggingFace repo  (fill in before publishing)",
                placeholder="username/my-fine-tuned-model",
                visible=False,
            )
            user_publish_token = gr.Textbox(
                label="HuggingFace Write token",
                placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                type="password",
                info="Get it at huggingface.co/settings/tokens → New token → Write",
                visible=False,
            )
            user_publish_out = gr.HTML("")

    # ── Evaluation screen ─────────────────────────────────────────────────────
    with gr.Group(visible=False) as eval_group:
        with gr.Row():
            gr.HTML('<span style="color:#8b949e;font-size:12px;padding:4px 0;">Evaluation</span>')
            back_eval_btn = gr.Button("← Home", size="sm", variant="secondary", scale=0)

        gr.HTML("""
        <div style="border-bottom:1px solid #30363d;padding-bottom:16px;margin-bottom:20px;">
          <h2 style="color:#e6edf3;margin:16px 0 6px;font-size:22px;font-weight:700;">
            Model Evaluation Dashboard
          </h2>
          <p style="color:#8b949e;font-size:13px;margin:0;">
            Compare your fine-tuned model against the base model across key performance metrics.
          </p>
        </div>
        """)

        with gr.Row():
            eval_run_dd  = gr.Dropdown(
                label="Training run",
                choices=list_eval_runs(),
                value=(list_eval_runs() or [None])[0],
                scale=5,
            )
            eval_ref_btn = gr.Button("↺ Refresh", size="sm", variant="secondary", scale=0)

        eval_no_data = gr.HTML(
            '<div style="text-align:center;color:#8b949e;padding:60px 0;font-size:14px;">'
            '📊 No completed training runs found.<br>'
            '<span style="font-size:12px;margin-top:6px;display:block;">'
            'Complete a training job first, then return here to view results.</span></div>',
            visible=not bool(list_eval_runs()),
        )

        with gr.Group(visible=bool(list_eval_runs())) as eval_charts_group:
            gr.HTML('<p style="color:#8b949e;font-size:11px;font-weight:600;'
                    'letter-spacing:0.06em;margin:4px 0 14px;">PERFORMANCE METRICS</p>')
            with gr.Row(equal_height=False):
                eval_loss_plot    = gr.Plot(show_label=False, scale=1)
                eval_ppl_plot     = gr.Plot(show_label=False, scale=1)
                eval_summary_plot = gr.Plot(show_label=False, scale=1)

            gr.HTML('<p style="color:#8b949e;font-size:11px;font-weight:600;'
                    'letter-spacing:0.06em;margin:22px 0 10px;">SUMMARY TABLE</p>')
            eval_table = gr.DataFrame(
                headers=["Metric", "Before fine-tuning", "After fine-tuning", "Change"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=False,
                row_count=(4, "fixed"),
            )

        def _load_eval(run_name):
            runs = list_eval_runs()
            has_data = bool(runs and run_name)
            loss_fig, ppl_fig, summary_fig, df = load_eval_run(run_name)
            return (
                loss_fig, ppl_fig, summary_fig, df,
                gr.HTML(visible=not has_data),
                gr.Group(visible=has_data),
            )

        eval_run_dd.change(
            _load_eval,
            inputs=eval_run_dd,
            outputs=[eval_loss_plot, eval_ppl_plot, eval_summary_plot,
                     eval_table, eval_no_data, eval_charts_group],
        )
        eval_ref_btn.click(
            lambda: gr.Dropdown(choices=list_eval_runs(),
                                value=(list_eval_runs() or [None])[0]),
            outputs=eval_run_dd,
        )

    # ── Landing buttons ───────────────────────────────────────────────────────
    # dev_group is empty — developer content lives outside (always rendered).
    # Hiding landing_group reveals the developer dashboard underneath.
    # user_group shows the simplified interface on top.

    # All group outputs: landing, dev_login, dev_group, user_group, dev_content, eval_group
    _outs = [landing_group, dev_login_group, dev_group, user_group, dev_content_group, eval_group]

    def _g(v): return gr.Group(visible=v)

    def go_to_login():
        return _g(False), _g(True),  _g(False), _g(False), _g(False), _g(False)

    def do_dev_login(username, password):
        if username.strip() == DEV_USERNAME and password.strip() == DEV_PASSWORD:
            return (_g(False), _g(False), _g(True),  _g(False), _g(True),  _g(False),
                    gr.HTML(""))
        else:
            return (_g(False), _g(True),  _g(False), _g(False), _g(False), _g(False),
                    gr.HTML('<div style="color:#f85149;font-size:13px;padding:8px 0;">Incorrect username or password.</div>'))

    def go_user():
        return _g(False), _g(False), _g(False), _g(True),  _g(False), _g(False)

    def go_home():
        return _g(True),  _g(False), _g(False), _g(False), _g(False), _g(False)

    def go_eval():
        runs = list_eval_runs()
        default = (runs or [None])[0]
        loss_fig, ppl_fig, summary_fig, df = load_eval_run(default)
        has_data = bool(default)
        return (
            _g(False), _g(False), _g(False), _g(False), _g(False), _g(True),
            gr.Dropdown(choices=runs, value=default),
            loss_fig, ppl_fig, summary_fig, df,
            gr.HTML(visible=not has_data),
            gr.Group(visible=has_data),
        )

    land_dev_btn.click(go_to_login, outputs=_outs)
    land_user_btn.click(go_user,    outputs=_outs)
    land_eval_btn.click(go_eval,    outputs=_outs + [
        eval_run_dd, eval_loss_plot, eval_ppl_plot, eval_summary_plot,
        eval_table, eval_no_data, eval_charts_group,
    ])

    dev_login_btn.click(do_dev_login,
                        inputs=[dev_login_user, dev_login_pass],
                        outputs=_outs + [dev_login_err])
    dev_login_user.submit(do_dev_login,
                          inputs=[dev_login_user, dev_login_pass],
                          outputs=_outs + [dev_login_err])
    dev_login_pass.submit(do_dev_login,
                          inputs=[dev_login_user, dev_login_pass],
                          outputs=_outs + [dev_login_err])

    dev_login_back_btn.click(go_home, outputs=_outs)
    back_dev_btn.click(go_home,       outputs=_outs)
    back_user_btn.click(go_home,      outputs=_outs)
    back_eval_btn.click(go_home,      outputs=_outs)

    # ── User mode screen navigation ───────────────────────────────────────────
    def show_screen(s1, s2a, s2b, s3):
        return (gr.Group(visible=s1), gr.Group(visible=s2a),
                gr.Group(visible=s2b), gr.Group(visible=s3))

    _all_screens = [user_screen1, user_screen2a, user_screen2b, user_screen3, user_screen4]

    user_qs_btn.click(lambda: show_screen4(False,True,False,False,False),  outputs=_all_screens)
    user_cm_btn.click(lambda: show_screen4(False,False,True,False,False),  outputs=_all_screens)
    user_qs_back_btn.click(lambda: show_screen4(True,False,False,False,False), outputs=_all_screens)
    user_cm_back_btn.click(lambda: show_screen4(True,False,False,False,False), outputs=_all_screens)

    # Show/hide steps box based on test mode checkbox
    user_qs_test_mode.change(
        lambda v: gr.Number(visible=v),
        inputs=user_qs_test_mode, outputs=user_qs_test_steps)

    # Show/hide file upload in Quick Start
    user_qs_dataset.change(
        lambda v: gr.File(visible="own data" in v.lower()),
        inputs=user_qs_dataset, outputs=user_qs_file)

    # Show/hide file upload in Custom Mix
    user_cm_dataset.change(
        lambda v: gr.File(visible=any("Upload" in x for x in v)),
        inputs=user_cm_dataset, outputs=user_cm_file)

    # ── User mode — Quick Start fine-tuning ───────────────────────────────────
    def build_progress_html(stage: str) -> str:
        stages = [
            ("prepare",  "Preparing your data"),
            ("configure","Setting up the training"),
            ("smoke",    "Running a quick check"),
            ("train",    "Training the model"),
            ("eval",     "Checking the results"),
            ("export",   "Saving your model"),
        ]
        order = [s[0] for s in stages]
        is_done = (stage == "done")
        current_idx = len(order) if is_done else (order.index(stage) if stage in order else 0)
        pct = 100 if is_done else int((current_idx / len(order)) * 100)

        # Circular SVG progress ring
        R = 54
        C = 2 * 3.14159 * R  # ~339.3
        offset = C * (1 - pct / 100)
        ring_color = "#3fb950" if is_done else "#58a6ff"
        pct_label = "Done!" if is_done else f"{pct}%"

        circle_svg = f"""
<svg viewBox="0 0 120 120" width="140" height="140" style="display:block;margin:0 auto;">
  <circle cx="60" cy="60" r="{R}" fill="none" stroke="#21262d" stroke-width="10"/>
  <circle cx="60" cy="60" r="{R}" fill="none" stroke="{ring_color}" stroke-width="10"
          stroke-dasharray="{C:.1f}" stroke-dashoffset="{offset:.1f}"
          stroke-linecap="round" transform="rotate(-90 60 60)"
          style="transition:stroke-dashoffset 0.6s ease;"/>
  <text x="60" y="56" text-anchor="middle" fill="{ring_color}"
        font-size="20" font-weight="700" font-family="monospace">{pct_label}</text>
  <text x="60" y="74" text-anchor="middle" fill="#8b949e"
        font-size="11" font-family="sans-serif">complete</text>
</svg>"""

        # Stage list
        rows = ""
        for i, (key, label) in enumerate(stages):
            if i < current_idx:
                icon = f'<span style="color:#3fb950;font-weight:700;font-size:15px;">✓</span>'
                col  = "#3fb950"
                extra = ""
            elif i == current_idx and not is_done:
                icon = f'<span style="color:#58a6ff;font-size:15px;animation:spin 1.2s linear infinite;display:inline-block;">◌</span>'
                col  = "#e6edf3"
                extra = '<span style="color:#58a6ff;font-size:11px;margin-left:8px;animation:pulse 1.5s ease infinite;">running...</span>'
            else:
                icon = f'<span style="color:#484f58;font-size:13px;">○</span>'
                col  = "#484f58"
                extra = ""

            rows += f"""
<div style="display:flex;align-items:center;gap:12px;padding:7px 0;
            border-bottom:1px solid #21262d;">
  <div style="width:20px;text-align:center;">{icon}</div>
  <span style="color:{col};font-size:14px;">{label}</span>{extra}
</div>"""

        title = "Training complete!" if is_done else "Training your model..."
        subtitle = "Taking you to chat..." if is_done else "This runs automatically. You will be taken to a chat screen when it's ready."

        return f"""
<style>
@keyframes spin  {{ to {{ transform:rotate(360deg); }} }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}
</style>
<div style="max-width:600px;margin:0 auto;padding:20px 0;">
  <h2 style="color:#e6edf3;margin:0 0 4px;font-size:22px;">{title}</h2>
  <p style="color:#8b949e;font-size:14px;margin:0 0 24px;">{subtitle}</p>
  <div style="display:flex;gap:32px;align-items:flex-start;background:#161b22;
              border:1px solid #30363d;border-radius:12px;padding:24px;">
    <div style="min-width:140px;">{circle_svg}</div>
    <div style="flex:1;">{rows}</div>
  </div>
</div>"""

    _user_last_model_path = {"path": None}   # shared state for inline chat
    _user_training_proc   = {"proc": None}   # track subprocess for cancellation

    def run_user_quickstart(dataset_choice, upload_file, model_name, test_mode=True, test_steps=50):
        dataset = "alpaca"
        model   = "qwen__qwen2.5-7b-instruct"
        test_steps = int(test_steps or 50)

        # For test mode: create a tiny 100-row dataset to skip preprocessing wait
        if test_mode:
            import json as _j
            src = ROOT / "datasets" / "alpaca" / "train.jsonl"
            tiny_dir = ROOT / "datasets" / "alpaca_tiny"
            tiny_dir.mkdir(exist_ok=True)
            if src.exists():
                rows = [_j.loads(l) for l in src.read_text().splitlines()[:100] if l.strip()]
                (tiny_dir / "train.jsonl").write_text("\n".join(_j.dumps(r) for r in rows))
                (tiny_dir / "val.jsonl").write_text("\n".join(_j.dumps(r) for r in rows[:10]))
                # minimal data card
                (ROOT / "data_cards" / "alpaca_tiny.json").write_text(
                    _j.dumps({"name": "alpaca_tiny", "train_rows": 100, "val_rows": 10}))
                dataset = "alpaca_tiny"

        yield build_progress_html("configure"), terminal_wrap("Generating recipe...")
        epochs = "1" if test_mode else "3"
        cfg_args = ["configure", f"--model={model}", f"--dataset={dataset}",
                    "--quantize=4bit", f"--epochs={epochs}", "--lr=2e-4"]
        out = ""
        for chunk in run_ft(cfg_args): out = chunk
        yield build_progress_html("smoke"), out

        configs = list_configs()
        if not configs:
            yield build_progress_html("smoke"), terminal_wrap('<span style="color:#f85149;">Config failed.</span>')
            return

        latest_cfg = f"configs/{configs[0]}"

        # Inject max_steps for test mode
        if test_mode:
            cfg_path = ROOT / "configs" / configs[0]
            if cfg_path.exists():
                txt = cfg_path.read_text()
                import re as _re
                if "max_steps" not in txt:
                    txt = _re.sub(r'^num_epochs:', f'max_steps: {test_steps}\nnum_epochs:', txt, flags=_re.MULTILINE)
                else:
                    txt = _re.sub(r'^max_steps:.*', f'max_steps: {test_steps}', txt, flags=_re.MULTILINE)
                cfg_path.write_text(txt)

        for chunk in run_ft(["smoke-test", latest_cfg, f"--steps={test_steps}"]):
            yield build_progress_html("smoke"), chunk

        yield build_progress_html("train"), terminal_wrap("Starting training...")
        for chunk in run_ft(["train", latest_cfg]):
            yield build_progress_html("train"), chunk

        yield build_progress_html("export"), terminal_wrap("Exporting...")
        new_ckpts = list_checkpoints()
        if new_ckpts:
            for chunk in run_ft(["export", f"checkpoints/{new_ckpts[0]}", "--merge", "--quantize=none"]):
                yield build_progress_html("export"), chunk
            exports = list_exports()
            if exports:
                _user_last_model_path["path"] = str(ROOT / "exports" / exports[0] / "merged")

        yield build_progress_html("done"), terminal_wrap(
            '<span style="color:#3fb950;font-weight:bold;font-size:16px;">Complete!</span>')


    def run_user_custommix(models_sel, datasets_sel, upload_file, model_name, test_steps=50, epochs=1):
        test_steps = int(test_steps or 50)
        epochs = int(epochs or 1)
        # Resolve model: match display label back to folder name
        model = "microsoft__phi-3.5-mini-instruct"  # default
        label = (models_sel or "").split("  ")[0].strip().lower().replace("-", "").replace(".", "")
        for m in list_models():
            if label in m.lower().replace("-", "").replace(".", "").replace("_", ""):
                model = m
                break

        # Resolve dataset: extract name before the row-count suffix
        dataset = "alpaca"
        for d in (datasets_sel or []):
            ds_name = d.split("  ")[0].strip()
            if ds_name and ds_name != "Upload my own data":
                dataset = ds_name
                break

        yield build_progress_html("configure"), terminal_wrap("Generating recipe...")
        cfg_args = ["configure", f"--model={model}", f"--dataset={dataset}",
                    "--quantize=4bit", f"--epochs={epochs}", "--lr=2e-4"]
        for chunk in run_ft(cfg_args): yield build_progress_html("configure"), chunk

        configs = list_configs()
        if not configs:
            yield build_progress_html("configure"), terminal_wrap('<span style="color:#f85149;">Config failed.</span>')
            return

        latest_cfg = f"configs/{configs[0]}"
        for chunk in run_ft(["smoke-test", latest_cfg, "--steps=20"]):
            yield build_progress_html("smoke"), chunk

        for chunk in run_ft(["train", latest_cfg]):
            yield build_progress_html("train"), chunk

        new_ckpts = list_checkpoints()
        if new_ckpts:
            for chunk in run_ft(["export", f"checkpoints/{new_ckpts[0]}", "--merge", "--quantize=none"]):
                yield build_progress_html("export"), chunk
            exports = list_exports()
            if exports:
                _user_last_model_path["path"] = str(ROOT / "exports" / exports[0] / "merged")

        yield build_progress_html("done"), terminal_wrap(
            '<span style="color:#3fb950;font-weight:bold;font-size:16px;">Complete!</span>')

    def run_user_custommix(models_sel, datasets_sel, upload_file, model_name, test_steps=50, epochs=1):
        test_steps = int(test_steps or 50)
        epochs = int(epochs or 1)
        model_map = {
            "Qwen2.5-7B-Instruct": "qwen__qwen2.5-7b-instruct",
            "Phi-3.5-mini-instruct": "phi-3.5-mini-instruct",
            "Qwen2.5-3B-Instruct": "qwen__qwen2.5-3b-instruct",
        }
        model_key = models_sel.split("  ")[0]
        model = model_map.get(model_key, "qwen__qwen2.5-7b-instruct")
        ds_names = []
        for d in (datasets_sel or []):
            if "alpaca" in d: ds_names.append("alpaca")
            elif "guanaco" in d: ds_names.append("guanaco-llama2-1k")
        if not ds_names: ds_names = ["alpaca"]
        dataset = ds_names[0]

        yield build_progress_html("configure"), terminal_wrap("Generating recipe..."), gr.Row(visible=False)
        cfg_args = ["configure", f"--model={model}", f"--dataset={dataset}",
                    "--quantize=4bit", f"--epochs={epochs}", "--lr=2e-4"]
        for chunk in run_ft(cfg_args):
            yield build_progress_html("configure"), chunk, gr.Row(visible=False)

        configs = list_configs()
        if not configs:
            yield build_progress_html("configure"), terminal_wrap('<span style="color:#f85149;">Config failed.</span>'), gr.Row(visible=False)
            return

        latest_cfg = f"configs/{configs[0]}"
        yield build_progress_html("smoke"), terminal_wrap("Running smoke test..."), gr.Row(visible=False)
        for chunk in run_ft(["smoke-test", latest_cfg, f"--steps={test_steps}"]):
            yield build_progress_html("smoke"), chunk, gr.Row(visible=False)

        yield build_progress_html("train"), terminal_wrap("Training started..."), gr.Row(visible=False)
        for chunk in run_ft(["train", latest_cfg]):
            yield build_progress_html("train"), chunk, gr.Row(visible=False)

        new_ckpts = list_checkpoints()
        if new_ckpts:
            yield build_progress_html("export"), terminal_wrap("Exporting..."), gr.Row(visible=False)
            for chunk in run_ft(["export", f"checkpoints/{new_ckpts[0]}", "--merge", "--quantize=none"]):
                yield build_progress_html("export"), chunk, gr.Row(visible=False)

        yield build_progress_html("export"), terminal_wrap(
            '<span style="color:#3fb950;font-weight:bold;font-size:16px;">Fine-tuning complete!</span><br>'
            '<span style="color:#8b949e;">Your model is ready. Click "Test in Chat" to try it.</span>'
        ), gr.Row(visible=True)

    def show_screen4(s1, s2a, s2b, s3, s4):
        return (gr.Group(visible=s1), gr.Group(visible=s2a),
                gr.Group(visible=s2b), gr.Group(visible=s3), gr.Group(visible=s4))

    def go_to_progress():
        return show_screen4(False, False, False, True, False)

    def show_cancel_btn():
        return gr.Button(visible=False), gr.Button(visible=True)

    def show_home_btn():
        return gr.Button(visible=True), gr.Button(visible=False)

    def go_to_chat():
        # Load model into user chat automatically
        model_path = _user_last_model_path.get("path")
        if model_path:
            load_model(model_path)
        return show_screen4(False, False, False, False, True)

    _btn_outs = [back_user_btn, cancel_train_btn]

    user_qs_start_btn.click(
        go_to_progress, outputs=_all_screens
    ).then(show_cancel_btn, outputs=_btn_outs
    ).then(
        run_user_quickstart,
        inputs=[user_qs_dataset, user_qs_file, user_qs_name, user_qs_test_mode, user_qs_test_steps],
        outputs=[user_progress_html, user_progress_out],
        show_progress=False,
    ).then(show_home_btn, outputs=_btn_outs
    ).then(go_to_chat, outputs=_all_screens)

    user_cm_start_btn.click(
        go_to_progress, outputs=_all_screens
    ).then(show_cancel_btn, outputs=_btn_outs
    ).then(
        run_user_custommix,
        inputs=[user_cm_model, user_cm_dataset, user_cm_file, user_cm_name, user_cm_test_steps, user_cm_epochs],
        outputs=[user_progress_html, user_progress_out],
        show_progress=False,
    ).then(show_home_btn, outputs=_btn_outs
    ).then(go_to_chat, outputs=_all_screens)

    user_new_btn.click(
        lambda: show_screen4(True,False,False,False,False), outputs=_all_screens)

    # Cancel training
    def cancel_training():
        import subprocess as _sp, signal as _sig
        try:
            _sp.run(["pkill", "-f", "axolotl.cli.train"], capture_output=True)
            _sp.run(["pkill", "-f", "ft_train.py"],       capture_output=True)
        except Exception:
            pass
        cancelled_html = build_progress_html("prepare").replace(
            "Training your model", "Training cancelled")
        return (
            cancelled_html,
            show_screen4(True, False, False, False, False)
            + (gr.Button(visible=True), gr.Button(visible=False))
        )

    def do_cancel():
        import subprocess as _sp
        _sp.run(["pkill", "-f", "axolotl.cli.train"], capture_output=True)
        _sp.run(["pkill", "-f", "ft_train.py"],       capture_output=True)
        return (
            gr.Button(visible=True),   # show Home
            gr.Button(visible=False),  # hide Cancel
            *show_screen4(True, False, False, False, False),
        )

    cancel_train_btn.click(
        do_cancel,
        outputs=[back_user_btn, cancel_train_btn] + _all_screens,
    )

    # Inline chat in Screen 4
    def user_chat_send(message, history):
        global _chat_model, _chat_tok, _chat_name
        if not message.strip():
            yield history or [], ""
            return
        history = list(history or [])
        if _chat_model is None:
            model_path = _user_last_model_path.get("path")
            # Validate path has actual model files; fallback to newest valid export if not
            def _is_valid_model(p):
                mp = Path(p) if p else None
                return mp and mp.exists() and (mp / "tokenizer.json").exists() and (mp / "config.json").exists()
            if not _is_valid_model(model_path):
                exports = list_exports()
                model_path = None
                for exp in exports:
                    candidate = ROOT / "exports" / exp / "merged"
                    if _is_valid_model(str(candidate)):
                        model_path = str(candidate)
                        break
            if not model_path:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": "No trained model found. Please complete the training step first."})
                yield history, ""
                return
            # Auto-load the model on first message
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "⏳ Loading your model, please wait (this takes ~30–60 seconds)..."})
            yield history, ""
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                mp = Path(model_path)
                _chat_tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True, local_files_only=True)
                try:
                    _chat_model = AutoModelForCausalLM.from_pretrained(
                        str(mp), device_map="auto", dtype=torch.bfloat16,
                        trust_remote_code=False, local_files_only=True)
                except Exception:
                    _chat_model = AutoModelForCausalLM.from_pretrained(
                        str(mp), device_map="auto", dtype=torch.bfloat16,
                        trust_remote_code=True, local_files_only=True)
                _chat_name = model_path
                history[-1]["content"] = "✓ Model loaded. Now answering your message..."
                yield history, ""
            except Exception as e:
                history[-1]["content"] = f"Failed to load model: {e}"
                yield history, ""
                return
        try:
            import torch
            msgs = [{"role": "system", "content": "You are a helpful assistant."}]
            for t in history:
                if isinstance(t, dict):
                    content = t.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(i.get("text", str(i)) for i in content)
                    msgs.append({"role": t["role"], "content": str(content)})
            msgs.append({"role": "user", "content": message})
            text = _chat_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids  = _chat_tok(text, return_tensors="pt").input_ids.to(_chat_model.device)
            _eos_ids = [_chat_tok.eos_token_id]
            _im_end = _chat_tok.convert_tokens_to_ids("<|im_end|>")
            if _im_end and _im_end != _chat_tok.eos_token_id:
                _eos_ids.append(_im_end)
            with torch.no_grad():
                out = _chat_model.generate(ids, max_new_tokens=256, do_sample=True,
                                           temperature=0.7, top_p=0.9,
                                           eos_token_id=_eos_ids,
                                           pad_token_id=_chat_tok.eos_token_id)
            reply = _chat_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False).strip()
            # Convert byte-fallback tokens like <0x0A> → actual characters (Qwen/sentencepiece models)
            import re as _re
            reply = _re.sub(r'<0x([0-9A-Fa-f]{2})>',
                            lambda m: bytes([int(m.group(1), 16)]).decode('utf-8', errors='replace'),
                            reply)
            reply = reply.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        except Exception as e:
            reply = f"Error: {e}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply})
        yield history, ""

    user_send_btn.click(user_chat_send,
                        inputs=[user_chat_input, user_chatbot],
                        outputs=[user_chatbot, user_chat_input], show_progress=False)
    user_chat_input.submit(user_chat_send,
                           inputs=[user_chat_input, user_chatbot],
                           outputs=[user_chatbot, user_chat_input], show_progress=False)
    user_clear_btn.click(lambda: [], outputs=user_chatbot)

    # Publish from user mode
    user_publish_btn.click(
        lambda: (gr.Textbox(visible=True), gr.Textbox(visible=True)),
        outputs=[user_publish_repo, user_publish_token])

    def do_user_publish(repo, token):
        if not repo or not repo.strip():
            yield terminal_wrap('<span style="color:#f85149;">Enter a HuggingFace repo name first.</span>')
            return
        if not token or not token.strip():
            yield terminal_wrap('<span style="color:#f85149;">Enter your HuggingFace Write token first.</span>')
            return
        # Find newest valid export
        best_export = None
        for exp in list_exports():
            candidate = ROOT / "exports" / exp / "merged"
            if candidate.exists() and (candidate / "tokenizer.json").exists() and (candidate / "config.json").exists():
                best_export = exp
                break
        if not best_export:
            yield terminal_wrap('<span style="color:#f85149;">No valid exported model found. Complete training first.</span>')
            return
        # Auto-fill MODEL_CARD.md placeholders so publish doesn't block
        import os as _os
        _os.environ["HF_TOKEN"] = token.strip()
        _os.environ["HUGGING_FACE_HUB_TOKEN"] = token.strip()
        card_path = ROOT / "exports" / best_export / "MODEL_CARD.md"
        if card_path.exists():
            txt = card_path.read_text()
            txt = txt.replace("{filled by user — describe what this model is for}",
                              "General-purpose instruction-following model fine-tuned by MKD Fine-Tuning Platform.")
            txt = txt.replace("{filled by user — describe known limitations}",
                              "May inherit limitations of the base model. Evaluate before production use.")
            txt = txt.replace("{repo-name}", repo.strip())
            txt = txt.replace("{optional}", "")
            card_path.write_text(txt)
        yield from run_ft(["publish", f"exports/{best_export}", "--hf-repo", repo.strip(),
                           "--acknowledge-contamination", "--accept-base-license"])

    user_publish_btn.click(do_user_publish, inputs=[user_publish_repo, user_publish_token],
                           outputs=user_publish_out, show_progress=False)

    with dev_content_group:
        # ── Header ────────────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);
                    border:1px solid #30363d;border-radius:12px;padding:24px 32px;margin-bottom:16px;">
          <div style="display:flex;align-items:center;gap:14px;">
            <div style="width:42px;height:42px;background:#1f6feb;border-radius:8px;
                        display:flex;align-items:center;justify-content:center;
                        font-size:18px;font-weight:700;color:#fff;letter-spacing:-1px;">MKD</div>
            <div>
              <h1 style="margin:0;font-size:24px;font-weight:700;color:#e6edf3;">
                MKD Fine-Tuning Platform
              </h1>
              <p style="margin:4px 0 0;color:#8b949e;font-size:14px;">
                Guided LLM fine-tuning · each step tells you what to do next
              </p>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <span style="background:#1f6feb;color:#e6edf3;padding:4px 12px;
                           border-radius:20px;font-size:12px;font-weight:500;">
                MKD Co., Ltd.
              </span>
            </div>
          </div>
        </div>
        """)
    
        # ── Pipeline + Stats + State row ──────────────────────────────────────────
        pipeline_html = gr.HTML(get_pipeline_html())
    
        with gr.Row(equal_height=True):
            with gr.Column(scale=3):
                stats_html = gr.HTML(get_stats_html())
            with gr.Column(scale=2):
                state_html = gr.HTML(get_state_html())
    
        with gr.Row():
            gpu_html = gr.HTML(get_gpu_html())
            refresh_all_btn = gr.Button("↺  Refresh dashboard", size="sm",
                                        variant="secondary", scale=0)
    
        dash_timer = gr.Timer(value=10)

        refresh_all_btn.click(
            lambda: (get_pipeline_html(), get_stats_html(), get_state_html(), get_gpu_html()),
            outputs=[pipeline_html, stats_html, state_html, gpu_html],
        )

        dash_timer.tick(
            lambda: (get_gpu_html(), get_pipeline_html(), get_stats_html(), get_state_html()),
            outputs=[gpu_html, pipeline_html, stats_html, state_html],
        )
    
        gr.HTML('<div style="border-top:1px solid #30363d;margin:8px 0 16px;"></div>')
    
        # ── Main tabs ─────────────────────────────────────────────────────────────
        with gr.Tabs():
    
            # ── 1 · Add Dataset ───────────────────────────────────────────────────
            with gr.Tab("1 · Dataset"):
                gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Add a dataset</h3>'
                        '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                        'Select a source below.</p>')
                # Source selector — radio avoids tab overflow
                ds_source_radio = gr.Radio(
                    choices=["HuggingFace", "Upload file", "Already on server", "AI Hub Korea"],
                    value="HuggingFace",
                    label="Where is your dataset?",
                    interactive=True,
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        # HuggingFace panel
                        with gr.Group(visible=True) as ds_panel_hf:
                            ds_src = gr.Textbox(label="HuggingFace dataset ID or URL",
                                                placeholder="tatsu-lab/alpaca")
                            ds_name_hf = gr.Textbox(label="Local name  (optional)", placeholder="alpaca")
                            add_ds_hf_btn = gr.Button("Download & Normalize", variant="primary")

                        # Upload panel
                        with gr.Group(visible=False) as ds_panel_upload:
                            ds_file = gr.File(label="Select a file",
                                              file_types=[".jsonl", ".json", ".csv", ".parquet"],
                                              file_count="single")
                            ds_name_file = gr.Textbox(label="Local name  (optional — defaults to filename)",
                                                       placeholder="leave blank to use filename")
                            add_ds_file_btn = gr.Button("Upload & Normalize", variant="primary")

                        # Server panel
                        with gr.Group(visible=False) as ds_panel_server:
                            existing_ds_cb = gr.CheckboxGroup(
                                label="Select datasets  (check one or more — multiple will be merged)",
                                choices=list_datasets())
                            with gr.Row():
                                existing_ds_ref = gr.Button("↺ Refresh", size="sm", variant="secondary")
                                existing_ds_use = gr.Button("Use selected", variant="primary")
                            existing_ds_info = gr.HTML(terminal_wrap(""))

                        # AI Hub panel
                        with gr.Group(visible=False) as ds_panel_aihub:
                            gr.HTML("""
                            <div style="background:#1a1a2e;border:1px solid #3d3d8f;border-radius:8px;
                                        padding:12px 16px;margin-bottom:12px;font-size:13px;">
                              <span style="color:#7b7bff;font-weight:600;">AI Hub Korea</span>
                              <span style="color:#484f58;font-size:12px;display:block;margin-top:4px;">
                                Get API key: aihub.or.kr → 마이페이지 → API 키 발급
                              </span>
                            </div>""")
                            aihub_id   = gr.Textbox(label="Dataset ID", placeholder="e.g. 71875")
                            aihub_key  = gr.Textbox(label="API Key", placeholder="Your AI Hub API key",
                                                     type="password")
                            aihub_name = gr.Textbox(label="Local name  (optional)",
                                                     placeholder="e.g. medical-knowledge")
                            aihub_btn  = gr.Button("Download from AI Hub", variant="primary")

                    with gr.Column(scale=3):
                        add_ds_out = gr.HTML(terminal_wrap(""), visible=True)

                # Show/hide panels + output panel based on radio selection
                def switch_ds_panel(choice):
                    show_output = choice in ("HuggingFace", "Upload file", "AI Hub Korea")
                    return (
                        gr.Group(visible=choice == "HuggingFace"),
                        gr.Group(visible=choice == "Upload file"),
                        gr.Group(visible=choice == "Already on server"),
                        gr.Group(visible=choice == "AI Hub Korea"),
                        gr.HTML(visible=show_output),
                    )
                ds_source_radio.change(
                    switch_ds_panel, inputs=ds_source_radio,
                    outputs=[ds_panel_hf, ds_panel_upload, ds_panel_server, ds_panel_aihub, add_ds_out]
                )
    
                def do_add_hf(src, name):
                    src = src.strip()
                    name = name.strip()
                    if not src:
                        yield terminal_wrap('<span style="color:#f85149;">Please enter a dataset ID or URL.</span>')
                        return
                    args = ["add-dataset", src] + (["--name", name] if name else [])
                    yield from run_ft(args)
    
                def do_upload(file, name):
                    if file is None:
                        yield terminal_wrap('<span style="color:#f85149;">No file selected.</span>')
                        return
                    derived = name.strip() or Path(file.name).stem.lower().replace(" ", "-")
                    yield from run_ft(["add-dataset", file.name, "--name", derived])
    
                def show_existing_ds(names):
                    if not names:
                        return terminal_wrap("")
                    html = ""
                    for name in names:
                        p = ROOT / "datasets" / name
                        files = [f.name for f in p.iterdir() if f.is_file()] if p.exists() else []
                        cleaned = (p / "train.jsonl").exists()
                        status_col = "#3fb950" if cleaned else "#d29922"
                        status_txt = "cleaned" if cleaned else "not cleaned yet"
                        html += (f'<span style="color:{status_col};font-weight:600;">{name}</span> '
                                 f'<span style="color:#8b949e;font-size:11px;">({status_txt})</span><br>'
                                 f'<span style="color:#484f58;font-size:11px;">{", ".join(files[:4])}</span><br><br>')
                    return terminal_wrap(html)
    
                def use_existing_ds(names):
                    if not names:
                        msg = terminal_wrap('<span style="color:#f85149;">Select at least one dataset.</span>')
                        return msg, terminal_wrap(""), gr.Dropdown()
                    if len(names) == 1:
                        name = names[0]
                        cleaned = (ROOT / "datasets" / name / "train.jsonl").exists()
                        next_tab = "4 · Configure" if cleaned else "2 · Clean"
                        rows_info = ""
                        try:
                            card = json.loads((ROOT / "data_cards" / f"{name}.json").read_text())
                            train = card.get("train_rows", "")
                            val   = card.get("val_rows", "")
                            if train:
                                rows_info = f'{train:,} train · {val:,} val rows'
                        except Exception:
                            pass
                        status_col = "#3fb950" if cleaned else "#d29922"
                        status_txt = "Cleaned and ready" if cleaned else "Not cleaned yet — go to 2 · Clean first"
                        info_card = f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;">
  <div style="font-size:16px;font-weight:700;color:#e6edf3;margin-bottom:8px;">{name}</div>
  <div style="color:{status_col};font-size:13px;margin-bottom:4px;">{status_txt}</div>
  {"<div style='color:#8b949e;font-size:12px;margin-bottom:12px;'>" + rows_info + "</div>" if rows_info else ""}
  <div style="color:#58a6ff;font-size:13px;font-weight:600;">Next step: go to <strong>{next_tab}</strong></div>
</div>"""
                        return info_card, terminal_wrap(""), gr.Dropdown(value=name)
                    # Multiple datasets — merge them
                    merged_name = "_".join(names)
                    merged_dir  = ROOT / "datasets" / merged_name
                    merged_dir.mkdir(parents=True, exist_ok=True)
                    import json as _j
                    rows = []
                    for name in names:
                        for fname in ["train.jsonl", "normalized.jsonl"]:
                            p = ROOT / "datasets" / name / fname
                            if p.exists():
                                rows += [_j.loads(l) for l in p.read_text().splitlines() if l.strip()]
                                break
                    if not rows:
                        return (terminal_wrap('<span style="color:#f85149;">No cleaned data found. Clean each dataset first.</span>'),
                                terminal_wrap(""), gr.Dropdown())
                    out = merged_dir / "normalized.jsonl"
                    with out.open("w") as f:
                        for r in rows:
                            f.write(_j.dumps(r, ensure_ascii=False) + "\n")
                    info_card = f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;">
  <div style="font-size:16px;font-weight:700;color:#e6edf3;margin-bottom:8px;">{merged_name}</div>
  <div style="color:#3fb950;font-size:13px;margin-bottom:4px;">Merged {len(names)} datasets</div>
  <div style="color:#8b949e;font-size:12px;margin-bottom:12px;">{len(rows):,} total rows</div>
  <div style="color:#58a6ff;font-size:13px;font-weight:600;">Next step: go to <strong>2 · Clean</strong></div>
</div>"""
                    return info_card, terminal_wrap(""), gr.Dropdown(value=merged_name)
    
                def do_aihub_download(dataset_id, api_key, ds_name):
                    """Download from AI Hub Korea, unzip, convert to ChatML."""
                    import zipfile, tempfile, subprocess as _sp

                    dataset_id = dataset_id.strip()
                    api_key    = api_key.strip()
                    ds_name    = ds_name.strip() or f"aihub-{dataset_id}"

                    if not dataset_id:
                        yield terminal_wrap('<span style="color:#f85149;">Enter a dataset ID.</span>')
                        return
                    if not api_key:
                        yield terminal_wrap('<span style="color:#f85149;">Enter your AI Hub API key.</span>')
                        return

                    yield terminal_wrap(f'<span style="color:#58a6ff;">Downloading dataset {dataset_id} from AI Hub...</span>')

                    # Download zip
                    raw_dir = ROOT / "datasets" / ds_name / "raw"
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    zip_path = raw_dir / f"{dataset_id}.zip"

                    dl_url = f"https://api.aihub.or.kr/down/0.5/{dataset_id}.do"
                    try:
                        import requests
                        r = requests.get(dl_url, headers={"apikey": api_key},
                                         stream=True, timeout=300)
                        if r.status_code != 200:
                            yield terminal_wrap(
                                f'<span style="color:#f85149;">Download failed: HTTP {r.status_code}<br>'
                                f'Check your API key and dataset ID.<br>'
                                f'Make sure you have access approved on aihub.or.kr</span>')
                            return

                        total = int(r.headers.get("content-length", 0))
                        downloaded = 0
                        with zip_path.open("wb") as f:
                            for chunk in r.iter_content(1024 * 1024):
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total:
                                    pct = int(downloaded / total * 100)
                                    yield terminal_wrap(
                                        f'<span style="color:#58a6ff;">Downloading... {pct}% ({downloaded//1024//1024} MB)</span>')
                    except Exception as e:
                        yield terminal_wrap(f'<span style="color:#f85149;">Download error: {e}</span>')
                        return

                    yield terminal_wrap(f'<span style="color:#3fb950;">Download complete. Extracting...</span>')

                    # Extract zip
                    extract_dir = raw_dir / "extracted"
                    extract_dir.mkdir(exist_ok=True)
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            zf.extractall(extract_dir)
                    except Exception as e:
                        yield terminal_wrap(f'<span style="color:#f85149;">Extraction failed: {e}</span>')
                        return

                    yield terminal_wrap('<span style="color:#3fb950;">Extracted. Converting to ChatML format...</span>')

                    # Find JSON files and convert AI Hub format → ChatML
                    import glob as _glob
                    json_files = list(extract_dir.rglob("*.json"))
                    if not json_files:
                        yield terminal_wrap('<span style="color:#f85149;">No JSON files found in zip.</span>')
                        return

                    rows = []
                    for jf in json_files:
                        try:
                            data = json.loads(jf.read_text(encoding="utf-8"))
                            # AI Hub medical Q&A formats
                            items = (data.get("데이터") or data.get("data") or
                                     data.get("지식베이스") or data.get("QA") or
                                     (data if isinstance(data, list) else []))
                            for item in items:
                                q = (item.get("질문") or item.get("question") or
                                     item.get("Q") or item.get("질의") or "")
                                a = (item.get("답변") or item.get("answer") or
                                     item.get("A") or item.get("응답") or "")
                                if q and a:
                                    rows.append({"messages": [
                                        {"role": "user",      "content": str(q)},
                                        {"role": "assistant", "content": str(a)},
                                    ]})
                        except Exception:
                            continue

                    if not rows:
                        yield terminal_wrap(
                            f'<span style="color:#d29922;">Converted 0 rows — AI Hub format may differ.<br>'
                            f'Files found: {[f.name for f in json_files[:3]]}<br>'
                            f'Use "Upload from computer" to manually process.</span>')
                        return

                    # Save normalized
                    out_dir = ROOT / "datasets" / ds_name
                    out_dir.mkdir(exist_ok=True)
                    norm = out_dir / "normalized.jsonl"
                    with norm.open("w", encoding="utf-8") as f:
                        for r in rows:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

                    # Write data card
                    card = {"name": ds_name, "source": f"aihub:{dataset_id}",
                            "rows_raw": len(rows), "detected_format": "aihub_medical"}
                    (ROOT / "data_cards").mkdir(exist_ok=True)
                    (ROOT / "data_cards" / f"{ds_name}.json").write_text(
                        json.dumps(card, indent=2))

                    yield terminal_wrap(
                        f'<span style="color:#3fb950;font-weight:bold;">Done! {len(rows):,} Q&A pairs converted.</span><br>'
                        f'<span style="color:#8b949e;">Dataset saved as "{ds_name}". Go to <strong>2 · Clean</strong> to prepare it.</span>'
                    ) + next_step_card("add-dataset")

                add_ds_hf_btn.click(do_add_hf, inputs=[ds_src, ds_name_hf],
                                    outputs=add_ds_out, show_progress=False)
                add_ds_file_btn.click(do_upload, inputs=[ds_file, ds_name_file],
                                      outputs=add_ds_out, show_progress=False)
                existing_ds_ref.click(lambda: gr.CheckboxGroup(choices=list_datasets()), outputs=existing_ds_cb)
                existing_ds_cb.change(show_existing_ds, inputs=existing_ds_cb, outputs=existing_ds_info)
                # existing_ds_use.click wired below after clean_dd is defined
                aihub_btn.click(do_aihub_download,
                                inputs=[aihub_id, aihub_key, aihub_name],
                                outputs=add_ds_out, show_progress=False)
    
            # ── 2 · Clean Dataset ─────────────────────────────────────────────────
            with gr.Tab("2 · Clean"):
                gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Clean, deduplicate & split</h3>'
                        '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                        'Dedup · PII scrub · length filter · train/val split · 100-row sample</p>')
                with gr.Row():
                    with gr.Column(scale=1):
                        clean_dd = gr.Dropdown(label="Dataset", choices=list_datasets(),
                                               allow_custom_value=True)
                        clean_max = gr.Slider(4096, 131072, value=32768, step=4096,
                                              label="Max chars per example")
                        clean_val = gr.Slider(0.005, 0.1, value=0.01, step=0.005,
                                              label="Validation ratio")
                        clean_nopii = gr.Checkbox(label="Skip PII scrub  (not recommended)")
                        with gr.Row():
                            clean_ref_btn = gr.Button("↺ Refresh", size="sm", variant="secondary")
                            clean_btn     = gr.Button(" Clean", variant="primary")
                    with gr.Column(scale=2):
                        clean_out = gr.HTML(terminal_wrap(""))
    
                gr.HTML('<div style="border-top:1px solid #30363d;margin:16px 0;"></div>')
                gr.HTML('<div style="border-top:1px solid #30363d;margin:16px 0 12px;"></div>')
                gr.HTML('<p style="color:#8b949e;font-size:11px;font-weight:600;'
                        'letter-spacing:0.06em;margin:0 0 10px;">DATA PREVIEW — auto-loads after cleaning</p>')
                with gr.Row():
                    with gr.Column(scale=2):
                        sample_out = gr.HTML(terminal_wrap(""))
                    with gr.Column(scale=2):
                        card_out = gr.HTML(terminal_wrap(""))

                def do_clean(n, mc, vr, nopii):
                    args = ["clean-dataset", n, "--max-chars", str(int(mc)), "--val-ratio", str(vr)]
                    if nopii: args.append("--no-pii")
                    yield from run_ft(args)

                def do_clean_and_preview(n, mc, vr, nopii):
                    # Stream cleaning output, then auto-load preview + data card
                    last = ""
                    for chunk in do_clean(n, mc, vr, nopii):
                        last = chunk
                        yield last, terminal_wrap(""), terminal_wrap("")
                    yield last, read_sample(n), read_data_card(n)

                clean_ref_btn.click(lambda: gr.Dropdown(choices=list_datasets()), outputs=clean_dd)
                clean_btn.click(
                    do_clean_and_preview,
                    inputs=[clean_dd, clean_max, clean_val, clean_nopii],
                    outputs=[clean_out, sample_out, card_out],
                    show_progress=False,
                )

                # Wire here so clean_dd is in scope — auto-fills Tab 2 when dataset selected in Tab 1
                existing_ds_use.click(use_existing_ds, inputs=existing_ds_cb,
                                      outputs=[existing_ds_info, add_ds_out, clean_dd])

            # ── 3 · Add Model ─────────────────────────────────────────────────────
            with gr.Tab("3 · Model"):
                gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Select a base model</h3>'
                        '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                        'Use a model already on the server, or download a new one.</p>')
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Tabs():
                            with gr.Tab("Server"):
                                gr.HTML('<p style="color:#8b949e;font-size:12px;margin:8px 0;">Models already downloaded on this server:</p>')
                                existing_model_dd = gr.Dropdown(
                                    label="Select model",
                                    choices=list_models(),
                                    allow_custom_value=False,
                                )
                                with gr.Row():
                                    model_server_ref = gr.Button("↺ Refresh", size="sm", variant="secondary")
                                    model_server_use = gr.Button("Use this model", variant="primary")
                                model_server_info = gr.HTML(terminal_wrap(""))
    
                            with gr.Tab("Download from HuggingFace"):
                                gr.HTML("""
                                <div style="background:#1f2937;border:1px solid #374151;border-radius:8px;
                                            padding:10px 14px;margin-bottom:12px;font-size:12px;">
                                  <span style="color:#3fb950;font-weight:600;">Open:</span>
                                  <code style="color:#58a6ff;margin:0 6px;">Qwen/Qwen2.5-7B-Instruct</code>
                                  <code style="color:#58a6ff;margin:0 6px;">microsoft/Phi-3.5-mini-instruct</code>
                                  <br><span style="color:#d29922;font-weight:600;">Gated:</span>
                                  <code style="color:#58a6ff;margin:0 6px;">meta-llama/Llama-3.1-8B-Instruct</code>
                                </div>
                                """)
                                model_src = gr.Textbox(label="HuggingFace model ID",
                                                        placeholder="Qwen/Qwen2.5-7B-Instruct")
                                model_lic = gr.Checkbox(label="Accept restricted license")
                                add_model_btn = gr.Button("Download Model", variant="primary")
    
                    with gr.Column(scale=2):
                        add_model_out = gr.HTML(terminal_wrap(""))
    
                def show_model_info(name):
                    if not name:
                        return terminal_wrap("")
                    p = ROOT / "models" / name / "MODEL_INFO.json"
                    if p.exists():
                        import json as _j
                        info = _j.loads(p.read_text())
                        html = f'<span style="color:#3fb950;font-weight:600;">Model ready: {info.get("model_id", name)}</span><br>'
                        html += f'Parameters: <span style="color:#58a6ff;">{info.get("params_billion", "?")}B</span><br>'
                        html += f'Architecture: {info.get("architecture", "?")}<br>'
                        html += f'Chat template: {"Yes" if info.get("chat_template_present") else "No"}<br>'
                        return terminal_wrap(html)
                    return terminal_wrap(f'<span style="color:#8b949e;">Model folder: {name}<br>No MODEL_INFO.json found.</span>')
    
                def use_server_model(name):
                    if not name:
                        return terminal_wrap('<span style="color:#f85149;">Select a model first.</span>'), gr.Dropdown()
                    return (
                        terminal_wrap(
                            f'<span style="color:#3fb950;font-weight:bold;">Model "{name}" selected.</span><br>'
                            f'<span style="color:#8b949e;">Auto-filled in 4 · Configure.</span>'
                        ) + next_step_card("add-model"),
                        gr.Dropdown(value=name),
                    )
    
                def do_add_model(s, a):
                    s = s.strip()
                    yield from run_ft(["add-model", s] + (["--accept-license"] if a else []))
    
                model_server_ref.click(lambda: gr.Dropdown(choices=list_models()), outputs=existing_model_dd)
                existing_model_dd.change(show_model_info, inputs=existing_model_dd, outputs=model_server_info)
                # model_server_use.click wired below after cfg_model is defined
                add_model_btn.click(do_add_model, inputs=[model_src, model_lic],
                                    outputs=add_model_out, show_progress=False)
    
            # ── 4 · Configure ─────────────────────────────────────────────────────
            with gr.Tab("4 · Configure"):
                gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Generate training recipe</h3>'
                        '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                        'Auto-detects GPU · sets micro-batch, optimizer, chat template · writes configs/run_&lt;ts&gt;.yaml</p>')
                with gr.Row():
                    with gr.Column(scale=1):
                        cfg_model = gr.Dropdown(label="Base model", choices=list_models(),
                                                 allow_custom_value=True)
                        cfg_ds    = gr.Dropdown(label="Dataset", choices=list_cleaned_datasets(),
                                                 allow_custom_value=True)
                        cfg_quant = gr.Radio(["4bit (QLoRA)", "8bit", "none (full LoRA)"],
                                              value="4bit (QLoRA)", label="Quantization")
                        with gr.Row():
                            cfg_lr     = gr.Number(value=2e-4, label="Learning rate", precision=6)
                            cfg_epochs = gr.Slider(1, 10, value=3, step=1, label="Epochs")
                        cfg_seq = gr.Slider(512, 8192, value=2048, step=512, label="Sequence length")
    
                        gr.HTML('<div style="border-top:1px solid #30363d;margin:12px 0 8px;"></div>')
                        cfg_wandb_enable = gr.Checkbox(
                            label="Enable Weights & Biases logging",
                            value=False,
                        )
                        with gr.Group(visible=False) as cfg_wandb_group:
                            cfg_wandb_project = gr.Textbox(
                                label="W&B Project", placeholder="mkd-finetuning")
                            cfg_wandb_entity  = gr.Textbox(
                                label="W&B Entity (username/team)", placeholder="mkd-hika")
    
                        cfg_wandb_enable.change(
                            lambda on: gr.Group(visible=on),
                            inputs=cfg_wandb_enable, outputs=cfg_wandb_group)
    
                        with gr.Row():
                            gr.Button("↺ Refresh", size="sm", variant="secondary").click(
                                lambda: (gr.Dropdown(choices=list_models()),
                                         gr.Dropdown(choices=list_cleaned_datasets())),
                                outputs=[cfg_model, cfg_ds])
                            cfg_btn = gr.Button("Generate", variant="primary")
    
                    with gr.Column(scale=2):
                        cfg_out  = gr.HTML(terminal_wrap(""))
                        cfg_yaml = gr.Code(label="Generated YAML", language="yaml",
                                           lines=22, interactive=False)
    
                def do_configure(model, ds, quant, lr, epochs, seq,
                                 wandb_on, wandb_project, wandb_entity):
                    q = quant.split(" ")[0]
                    args = [
                        "configure", "--model", model, "--dataset", ds,
                        "--quantize", q, "--lr", str(lr),
                        "--epochs", str(int(epochs)), "--seq-len", str(int(seq)),
                    ]
                    if wandb_on and wandb_project.strip():
                        args += ["--wandb-project", wandb_project.strip()]
                    if wandb_on and wandb_entity.strip():
                        args += ["--wandb-entity", wandb_entity.strip()]
    
                    output_html = terminal_wrap("")
                    for chunk in run_ft(args):
                        output_html = chunk
                        yield output_html, ""
                    cfgs = list_configs()
                    yaml_txt = read_config_file(cfgs[0]) if cfgs else ""
                    yield output_html, yaml_txt
    
                cfg_btn.click(do_configure,
                              inputs=[cfg_model, cfg_ds, cfg_quant, cfg_lr, cfg_epochs, cfg_seq,
                                      cfg_wandb_enable, cfg_wandb_project, cfg_wandb_entity],
                              outputs=[cfg_out, cfg_yaml], show_progress=False)

                # Cross-tab: Tab 3 "Use this model" → auto-fill cfg_model in Tab 4
                model_server_use.click(use_server_model,
                                       inputs=existing_model_dd,
                                       outputs=[model_server_info, cfg_model])

                # Cross-tab: Tab 2 "Clean" done → auto-fill cfg_ds in Tab 4
                clean_btn.click(
                    lambda ds: gr.Dropdown(value=ds),
                    inputs=clean_dd, outputs=cfg_ds,
                )

                # Cross-tab: Tab 1 "Use selected" → auto-fill cfg_ds in Tab 4
                existing_ds_use.click(
                    lambda ds_name: gr.Dropdown(value=ds_name),
                    inputs=clean_dd, outputs=cfg_ds,
                )

            # ── 5 · Train ─────────────────────────────────────────────────────────
            with gr.Tab("5 · Train"):
                def _recipe_label_html():
                    cfg = list_configs()[0] if list_configs() else "none — run Configure first"
                    return (f'<div style="color:#8b949e;font-size:13px;margin-bottom:12px;">'
                            f'Using recipe: <span style="color:#58a6ff;font-weight:600;">{cfg}</span></div>')

                train_recipe_label = gr.HTML(value=_recipe_label_html())

                with gr.Row():
                    smoke_steps = gr.Number(
                        label="Smoke test steps", value=50, precision=0,
                        minimum=10, maximum=200,
                        info="10 ≈ 30 sec · 50 ≈ 2 min · 200 ≈ 8 min", scale=1,
                    )
                    train_max_steps = gr.Number(
                        label="Max training steps (0 = run all epochs)",
                        value=0, precision=0, minimum=0, scale=1,
                        info="e.g. 200 for a quick test · 0 = use epochs from Configure",
                    )

                with gr.Row():
                    smoke_btn      = gr.Button("Smoke Test",         variant="secondary", scale=1)
                    train_btn      = gr.Button("Start Full Training", variant="primary",   scale=2)
                    dev_cancel_btn = gr.Button("Cancel",              variant="stop",      scale=0, visible=False)

                train_out = gr.HTML(terminal_wrap(""))

                def _latest_config():
                    cfgs = list_configs()
                    return cfgs[0] if cfgs else ""

                def do_smoke(steps):
                    c = _latest_config()
                    if not c:
                        yield gr.Button(visible=False), terminal_wrap(
                            '<span style="color:#f85149;">No recipe found — run Configure first.</span>')
                        return
                    yield gr.Button(visible=True), terminal_wrap(f"Running smoke test with {c}...")
                    for chunk in run_ft(["smoke-test", f"configs/{c}", "--steps", str(int(steps))]):
                        yield gr.Button(visible=True), chunk
                    yield gr.Button(visible=False), chunk

                def do_train(max_steps):
                    c = _latest_config()
                    if not c:
                        yield gr.Button(visible=False), terminal_wrap(
                            '<span style="color:#f85149;">No recipe found — run Configure first.</span>')
                        return
                    cmd = ["train", f"configs/{c}"]
                    if int(max_steps or 0) > 0:
                        cmd += ["--max-steps", str(int(max_steps))]
                    chunk = ""
                    yield gr.Button(visible=True), terminal_wrap(f"Starting training with {c}...")
                    for chunk in run_ft(cmd):
                        yield gr.Button(visible=True), chunk
                    yield gr.Button(visible=False), chunk

                def dev_cancel():
                    import subprocess as _sp
                    _sp.run(["pkill", "-f", "axolotl.cli.train"], capture_output=True)
                    _sp.run(["pkill", "-f", "ft_train.py"],       capture_output=True)
                    _sp.run(["pkill", "-f", "ft_smoke_test.py"],  capture_output=True)
                    return gr.Button(visible=False), terminal_wrap(
                        '<span style="color:#f85149;font-weight:600;">Training cancelled.</span>')

                smoke_btn.click(do_smoke, inputs=[smoke_steps],
                                outputs=[dev_cancel_btn, train_out], show_progress=False)
                train_btn.click(do_train, inputs=[train_max_steps],
                                outputs=[dev_cancel_btn, train_out], show_progress=False)
                dev_cancel_btn.click(dev_cancel, outputs=[dev_cancel_btn, train_out])
    
            # ── 6 · Eval ──────────────────────────────────────────────────────────
            with gr.Tab("6 · Eval"):
                gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Evaluate the fine-tuned model</h3>'
                        '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                        'Standard benchmarks · compare vs base model · contamination check</p>')
                with gr.Row():
                    eval_dd  = gr.Dropdown(label="Checkpoint", choices=list_checkpoints(),
                                            allow_custom_value=True, scale=3)
                    eval_ref = gr.Button("↺", size="sm", variant="secondary", scale=0)
                with gr.Row():
                    eval_skip    = gr.Checkbox(label="Skip harness evals  (contamination only — fast)")
                    eval_compare = gr.Checkbox(label="Compare vs base model", value=True)
                eval_limit = gr.Slider(
                    minimum=50, maximum=1000, value=100, step=50,
                    label="Examples per benchmark  (lower = faster, less accurate · full dataset = 1000+)",
                )
                gr.HTML('<p style="color:#8b949e;font-size:12px;margin:-8px 0 12px;">'
                        '100 examples ≈ 5 min · 500 examples ≈ 20 min · full dataset ≈ 60+ min</p>')
                eval_btn = gr.Button("Run Evaluation", variant="primary")
                eval_out = gr.HTML(terminal_wrap(""))
    
                gr.HTML('<div style="border-top:1px solid #30363d;margin:16px 0;"></div>')
                gr.HTML('<h4 style="color:#c9d1d9;margin:0 0 10px;">Benchmark comparison — Fine-tuned vs Base</h4>')
    
                eval_chart = gr.BarPlot(
                    value=pd.DataFrame({"Benchmark": [], "Score": [], "Model": []}),
                    x="Benchmark", y="Score", color="Model",
                    title="Fine-tuned vs Base Model",
                    y_lim=[0, 1],
                    height=340,
                    color_map={"Fine-tuned": "#3fb950", "Base model": "#58a6ff"},
                )
    
                gr.HTML('<h4 style="color:#c9d1d9;margin:12px 0 8px;">Summary</h4>')
                eval_summary = gr.HTML(terminal_wrap(""))
    
                def get_comparison_chart(run_id: str) -> pd.DataFrame:
                    """Build comparison DataFrame from saved eval results."""
                    result_dir = ROOT / "eval_results" / run_id
                    rows = []
                    for fname in result_dir.glob("*.json"):
                        if "contamination" in fname.name or "env" in fname.name:
                            continue
                        try:
                            data = json.loads(fname.read_text())
                            results = data.get("results", {})
                            for task, metrics in results.items():
                                for key, val in metrics.items():
                                    if key in ("acc,none", "acc_norm,none", "exact_match,none"):
                                        if isinstance(val, (int, float)):
                                            label = fname.stem.replace("_", " ").title()
                                            rows.append({
                                                "Benchmark": task.replace("_", " ").upper()[:12],
                                                "Score": round(float(val), 3),
                                                "Model": "Fine-tuned",
                                            })
                        except Exception:
                            pass
    
                    # Load base model results if available
                    base_file = result_dir / "base_model_results.json"
                    if base_file.exists():
                        try:
                            data = json.loads(base_file.read_text())
                            for task, metrics in data.get("results", {}).items():
                                for key, val in metrics.items():
                                    if key in ("acc,none", "acc_norm,none", "exact_match,none"):
                                        if isinstance(val, (int, float)):
                                            rows.append({
                                                "Benchmark": task.replace("_", " ").upper()[:12],
                                                "Score": round(float(val), 3),
                                                "Model": "Base model",
                                            })
                        except Exception:
                            pass
    
                    if not rows:
                        return pd.DataFrame({"Benchmark": [], "Score": [], "Model": []})
                    return pd.DataFrame(rows)
    
                def do_eval(ckpt, skip, compare, limit):
                    args = ["eval", f"checkpoints/{ckpt}", "--limit", str(int(limit))]
                    if skip:
                        args.append("--skip-harness")
                    if compare:
                        args.append("--compare-base")
                    for chunk in run_ft(args):
                        yield chunk, pd.DataFrame({"Benchmark": [], "Score": [], "Model": []}), terminal_wrap("")
                    chart_df = get_comparison_chart(ckpt)
                    summary  = read_eval_summary(ckpt)
                    yield chunk, chart_df, summary
    
                eval_ref.click(lambda: gr.Dropdown(choices=list_checkpoints()), outputs=eval_dd)
                eval_btn.click(do_eval, inputs=[eval_dd, eval_skip, eval_compare, eval_limit],
                               outputs=[eval_out, eval_chart, eval_summary], show_progress=False)
                eval_dd.change(
                    lambda ckpt: (get_comparison_chart(ckpt), read_eval_summary(ckpt)),
                    inputs=eval_dd, outputs=[eval_chart, eval_summary]
                )
    
            # ── 7 · Export & Publish ──────────────────────────────────────────────
            with gr.Tab("7 · Export & Publish"):
                gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Merge · Quantize · Publish</h3>'
                        '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                        'Merge LoRA into base · AWQ/GGUF quantize · edit model card · push to HuggingFace</p>')
    
                with gr.Tabs():
                    # Export
                    with gr.Tab("Export"):
                        with gr.Row():
                            exp_dd  = gr.Dropdown(label="Checkpoint", choices=list_checkpoints(),
                                                   allow_custom_value=True, scale=3)
                            exp_ref = gr.Button("↺", size="sm", variant="secondary", scale=0)
                        with gr.Row():
                            exp_quant = gr.Radio(["awq", "none"], value="awq",
                                                  label="Quantization format")
                            exp_gguf  = gr.Checkbox(label="Also export GGUF  (for llama.cpp)")
                        exp_btn = gr.Button(" Export Model", variant="primary")
                        exp_out = gr.HTML(terminal_wrap(""))
    
                        def do_export(ckpt, q, gguf):
                            args = ["export", f"checkpoints/{ckpt}", "--merge", "--quantize", q]
                            if gguf: args.append("--gguf")
                            yield from run_ft(args)
    
                        exp_ref.click(lambda: gr.Dropdown(choices=list_checkpoints()), outputs=exp_dd)
                        exp_btn.click(do_export, inputs=[exp_dd, exp_quant, exp_gguf],
                                      outputs=exp_out, show_progress=False)
    
                    # Model Card
                    with gr.Tab("Model Card"):
                        with gr.Row():
                            card_dd  = gr.Dropdown(label="Export", choices=list_exports(),
                                                    allow_custom_value=True, scale=3)
                            card_ref = gr.Button("↺", size="sm", variant="secondary", scale=0)
                            card_load = gr.Button(" Load", size="sm", variant="secondary", scale=0)
                        card_editor = gr.Textbox(
                            label="MODEL_CARD.md  —  fill in all {placeholders} before publishing",
                            lines=22, placeholder="Select an export and click Load...")
                        with gr.Row():
                            card_save = gr.Button(" Save", variant="primary", scale=0)
                            card_msg  = gr.Textbox(label="", lines=1, interactive=False, scale=2)
    
                        card_ref.click(lambda: gr.Dropdown(choices=list_exports()), outputs=card_dd)
                        card_load.click(read_model_card, inputs=card_dd, outputs=card_editor)
                        card_save.click(save_model_card, inputs=[card_dd, card_editor],
                                        outputs=card_msg)
    
                    # Publish
                    with gr.Tab("Publish"):
                        with gr.Row():
                            pub_dd  = gr.Dropdown(label="Export", choices=list_exports(),
                                                   allow_custom_value=True, scale=3)
                            pub_ref = gr.Button("↺", size="sm", variant="secondary", scale=0)
                        hf_repo = gr.Textbox(label="HuggingFace repo",
                                              placeholder="mkd-hika/qwen2.5-7b-alpaca")
                        hf_token = gr.Textbox(label="HuggingFace Write token",
                                               placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                                               type="password",
                                               info="Get it at huggingface.co/settings/tokens → New token → Write")
                        gh_repo = gr.Textbox(label="GitHub repo  (optional)",
                                              placeholder="mkd-hika/qwen2.5-7b-alpaca")
                        gr.HTML("""
                        <div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
                                    padding:10px 16px;font-size:12px;color:#8b949e;margin-bottom:8px;">
                          ℹ Publish will be blocked if MODEL_CARD.md has unfilled placeholders
                          or contamination ratio &gt; 0.5%.
                        </div>
                        """)
                        with gr.Row():
                            pub_ack = gr.Checkbox(label="Acknowledge contamination")
                            pub_lic = gr.Checkbox(label="Accept base model license")
                        pub_btn = gr.Button(" Publish to HuggingFace", variant="primary")
                        pub_out = gr.HTML(terminal_wrap(""))
    
                        pub_ref.click(lambda: gr.Dropdown(choices=list_exports()), outputs=pub_dd)
                        def do_publish(exp, hf, token, gh, ack, lic):
                            if not hf or not hf.strip():
                                yield terminal_wrap('<span style="color:#f85149;font-weight:600;">✗ HuggingFace repo name is required — e.g. mkd-hika/qwen2.5-7b-alpaca</span>')
                                return
                            if "/" not in hf.strip():
                                yield terminal_wrap('<span style="color:#f85149;font-weight:600;">✗ Repo must be in format: username/repo-name</span>')
                                return
                            if not token or not token.strip():
                                yield terminal_wrap('<span style="color:#f85149;font-weight:600;">✗ HuggingFace Write token is required — get it at huggingface.co/settings/tokens</span>')
                                return
                            import os as _os
                            _os.environ["HF_TOKEN"] = token.strip()
                            _os.environ["HUGGING_FACE_HUB_TOKEN"] = token.strip()
                            args = ["publish", f"exports/{exp}", "--hf-repo", hf.strip()]
                            if gh.strip(): args += ["--github-repo", gh.strip()]
                            if ack: args.append("--acknowledge-contamination")
                            if lic: args.append("--accept-base-license")
                            yield from run_ft(args)

                        pub_btn.click(do_publish, inputs=[pub_dd, hf_repo, hf_token, gh_repo, pub_ack, pub_lic],
                                      outputs=pub_out, show_progress=False)
    
            # ── 8 · Chat ──────────────────────────────────────────────────────────
            with gr.Tab("8 · Chat"):
                gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Test your model</h3>'
                        '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                        'Load any checkpoint or base model and chat with it directly.</p>')
                with gr.Row():
                    with gr.Column(scale=1):
                        def list_chat_models():
                            opts = []
                            # Merged exports first (fine-tuned, ready to use)
                            exp = ROOT / "exports"
                            if exp.exists():
                                for d in sorted(exp.iterdir()):
                                    m = d / "merged"
                                    if m.exists() and any(m.glob("model*.safetensors")):
                                        opts.append(f"exports/{d.name}/merged  [fine-tuned]")
                            # Base models
                            opts += [f"models/{n}  [base]" for n in list_models()]
                            # LoRA checkpoints
                            opts += [f"checkpoints/{n}  [adapter]" for n in list_checkpoints()]
                            return opts
    
                        def resolve_chat_path(selection: str) -> str:
                            """Strip label suffix and resolve to full path."""
                            path = selection.split("  [")[0].strip()
                            full = ROOT / path
                            return str(full) if full.exists() else path
    
                        chat_model_dd = gr.Dropdown(
                            label="Model or checkpoint",
                            choices=list_chat_models(),
                            allow_custom_value=True,
                        )
                        gr.Button("↺ Refresh", size="sm", variant="secondary").click(
                            lambda: gr.Dropdown(choices=list_chat_models()),
                            outputs=chat_model_dd)
                        chat_system = gr.Textbox(
                            label="System prompt",
                            value="You are a helpful assistant.",
                            lines=3,
                        )
                        chat_max_tokens = gr.Slider(64, 1024, value=256, step=64,
                                                     label="Max new tokens")
                        with gr.Row():
                            chat_load_btn  = gr.Button("Load model", variant="primary")
                            chat_unload_btn = gr.Button("Unload", variant="secondary")
                        chat_status = gr.HTML(
                            '<div style="color:#8b949e;font-size:12px;padding:8px 0;">No model loaded.</div>')
    
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(height=420)
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Type your message...",
                                label="", scale=4, lines=1,
                            )
                            chat_send  = gr.Button("Send", variant="primary", scale=0)
                        chat_clear = gr.Button("Clear conversation", size="sm", variant="secondary")
    
                def load_model(model_name):
                    global _chat_model, _chat_tok, _chat_name
                    if not model_name:
                        return '<div style="color:#f85149;padding:8px;">Select a model first.</div>'
                    # Unload previous
                    if _chat_model is not None:
                        del _chat_model
                        _chat_model = None
                        import gc; gc.collect()
                        try:
                            import torch; torch.cuda.empty_cache()
                        except Exception:
                            pass
    
                    # Resolve the path (strips label suffix like "  [fine-tuned]")
                    resolved = resolve_chat_path(model_name)
                    model_path = Path(resolved)
                    if not model_path.exists():
                        model_path = ROOT / resolved
                    if not model_path.exists():
                        return f'<div style="color:#f85149;padding:8px;">Path not found: {resolved}</div>'
    
                    try:
                        from transformers import AutoTokenizer, AutoModelForCausalLM
                        import torch
                        _chat_tok = AutoTokenizer.from_pretrained(
                            str(model_path), trust_remote_code=True, local_files_only=True)
                        try:
                            _chat_model = AutoModelForCausalLM.from_pretrained(
                                str(model_path), device_map="auto",
                                dtype=torch.bfloat16, trust_remote_code=False,
                                local_files_only=True)
                        except Exception:
                            _chat_model = AutoModelForCausalLM.from_pretrained(
                                str(model_path), device_map="auto",
                                dtype=torch.bfloat16, trust_remote_code=True,
                                local_files_only=True)
                        _chat_name = model_name
                        label = model_name.split("  [")[0]
                        return f'<div style="color:#3fb950;font-weight:600;padding:8px;">Loaded: {label}</div>'
                    except Exception as e:
                        return f'<div style="color:#f85149;padding:8px;">Failed: {e}</div>'
    
                def unload_model():
                    global _chat_model, _chat_tok, _chat_name
                    if _chat_model is not None:
                        del _chat_model; _chat_model = None
                        del _chat_tok;   _chat_tok = None
                        _chat_name = None
                        import gc; gc.collect()
                        try:
                            import torch; torch.cuda.empty_cache()
                        except Exception:
                            pass
                    return '<div style="color:#8b949e;padding:8px;">Model unloaded. GPU memory freed.</div>'
    
                def chat_respond(message, history, system_prompt, max_tokens):
                    global _chat_model, _chat_tok
                    if not message.strip():
                        yield history or [], ""
                        return
    
                    history = list(history or [])
    
                    if _chat_model is None:
                        history.append({"role": "user",      "content": message})
                        history.append({"role": "assistant", "content": "No model loaded. Please load a model first."})
                        yield history, ""
                        return
    
                    try:
                        import torch
                        # Build messages for the model
                        def extract_text(content) -> str:
                            """Gradio 6 stores content as str or [{'text':...,'type':'text'}]."""
                            if isinstance(content, str):
                                return content
                            if isinstance(content, list):
                                parts = []
                                for item in content:
                                    if isinstance(item, dict):
                                        parts.append(item.get("text", str(item)))
                                    else:
                                        parts.append(str(item))
                                return " ".join(parts)
                            return str(content)
    
                        msgs = [{"role": "system", "content": system_prompt}]
                        for turn in history:
                            if isinstance(turn, dict):
                                msgs.append({
                                    "role": turn.get("role", "user"),
                                    "content": extract_text(turn.get("content", "")),
                                })
                        msgs.append({"role": "user", "content": message})
    
                        text = _chat_tok.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True)
                        enc = _chat_tok(text, return_tensors="pt").to(_chat_model.device)
                        ids = enc.input_ids
                        attn_mask = enc.attention_mask
                        _eos_ids = [_chat_tok.eos_token_id]
                        _im_end = _chat_tok.convert_tokens_to_ids("<|im_end|>")
                        if _im_end and _im_end != _chat_tok.eos_token_id:
                            _eos_ids.append(_im_end)
                        with torch.no_grad():
                            out = _chat_model.generate(
                                ids, attention_mask=attn_mask,
                                max_new_tokens=max_tokens,
                                do_sample=True, temperature=0.7, top_p=0.9,
                                eos_token_id=_eos_ids,
                                pad_token_id=_chat_tok.eos_token_id,
                            )
                        reply = _chat_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False).strip()
                        import re as _re
                        reply = _re.sub(r'<0x([0-9A-Fa-f]{2})>',
                                        lambda m: bytes([int(m.group(1), 16)]).decode('utf-8', errors='replace'),
                                        reply)
                        reply = reply.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
                    except Exception as e:
                        reply = f"Error: {e}"

                    history.append({"role": "user",      "content": message})
                    history.append({"role": "assistant", "content": reply})
                    yield history, ""
    
                chat_load_btn.click(load_model, inputs=chat_model_dd, outputs=chat_status)
                chat_unload_btn.click(unload_model, outputs=chat_status)
                chat_send.click(chat_respond,
                                inputs=[chat_input, chatbot, chat_system, chat_max_tokens],
                                outputs=[chatbot, chat_input], show_progress=False)
                chat_input.submit(chat_respond,
                                  inputs=[chat_input, chatbot, chat_system, chat_max_tokens],
                                  outputs=[chatbot, chat_input], show_progress=False)
                chat_clear.click(lambda: [], outputs=chatbot)
    
        # ── Footer ────────────────────────────────────────────────────────────────
        gr.HTML("""
    <div style="text-align:center;color:#484f58;font-size:12px;margin-top:24px;padding:16px;
                border-top:1px solid #21262d;">
      MKD Co., Ltd. · Fine-Tuning Platform · Built with Axolotl + Gradio
    </div>
    """)


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",  type=int, default=7860)
    ap.add_argument("--host",  default="0.0.0.0")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    print(f"\n   MKD Fine-Tuning Platform")
    print(f" Project : {ROOT}")
    print(f" URL     : http://localhost:{args.port}\n")

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0d1117",
        body_background_fill_dark="#0d1117",
        block_background_fill="#161b22",
        block_background_fill_dark="#161b22",
        block_border_color="#30363d",
        block_border_color_dark="#30363d",
        input_background_fill="#161b22",
        input_background_fill_dark="#161b22",
        input_border_color="#30363d",
        input_border_color_dark="#30363d",
        body_text_color="#e6edf3",
        body_text_color_dark="#e6edf3",
        body_text_color_subdued="#8b949e",
        body_text_color_subdued_dark="#8b949e",
        input_placeholder_color="#6e7681",
        input_placeholder_color_dark="#6e7681",
        button_primary_background_fill="#238636",
        button_primary_background_fill_dark="#238636",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#21262d",
        button_secondary_background_fill_dark="#21262d",
        button_secondary_text_color="#c9d1d9",
        button_secondary_text_color_dark="#c9d1d9",
    )

    app.queue(max_size=20)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=theme,
        css=CSS,
        max_threads=10,
    )
