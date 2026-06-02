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

import gradio as gr
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
            "action": 'Go to <b>1 · Dataset</b> → paste a HuggingFace ID (e.g. tatsu-lab/alpaca) → click <b>Download & Normalize</b>',
        },
        {
            "num": 2, "label": "Clean Dataset",
            "done": bool(list_cleaned_datasets()),
            "tab": "2 · Clean",
            "action": 'Go to <b>2 · Clean</b> → select your dataset → click <b>Clean</b>',
        },
        {
            "num": 3, "label": "Add Model",
            "done": bool(list_models()),
            "tab": "3 · Model",
            "action": 'Go to <b>3 · Model</b> → select from server or paste a HuggingFace model ID → click <b>Download Model</b>',
        },
        {
            "num": 4, "label": "Configure",
            "done": bool(list_configs()),
            "tab": "4 · Configure",
            "action": 'Go to <b>4 · Configure</b> → select model + dataset → click <b>Generate</b>',
        },
        {
            "num": 5, "label": "Train",
            "done": any(
                any((ROOT/"checkpoints"/d).glob("checkpoint-*"))
                for d in list_checkpoints()
            ) if list_checkpoints() else False,
            "tab": "5 · Train",
            "action": 'Go to <b>5 · Train</b> → select recipe → click <b>Smoke Test</b> first, then <b>Start Full Training</b>',
        },
        {
            "num": 6, "label": "Evaluate",
            "done": any(
                (ROOT/"eval_results"/d/"summary.md").exists()
                for d in list_checkpoints()
            ) if list_checkpoints() else False,
            "tab": "6 · Eval",
            "action": 'Go to <b>6 · Eval</b> → select checkpoint → uncheck "Skip harness evals" → click <b>Run Evaluation</b>',
        },
        {
            "num": 7, "label": "Export",
            "done": bool(list_exports()) and any(
                any((ROOT/"exports"/d/"merged").glob("model*.safetensors"))
                for d in list_exports()
            ) if list_exports() else False,
            "tab": "7 · Export & Publish",
            "action": 'Go to <b>7 · Export & Publish</b> → Export tab → select checkpoint → set quantization to <b>none</b> → click <b>Export Model</b>',
        },
        {
            "num": 8, "label": "Publish",
            "done": False,  # can't auto-detect publish
            "tab": "7 · Export & Publish",
            "action": 'Go to <b>7 · Export & Publish</b> → Model Card tab → fill placeholders → Publish tab → enter HF repo → click <b>Publish to HuggingFace</b>',
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
<div style="display:flex;align-items:flex-start;gap:10px;padding:8px 10px;
            border-radius:6px;background:{bg};margin-bottom:4px;">
  <div style="margin-top:2px;min-width:20px;">{icon}</div>
  <div style="flex:1;">
    <div style="color:{color};font-size:13px;font-weight:600;">{s["label"]}</div>
    {"<div style='color:#8b949e;font-size:12px;margin-top:2px;'>" + s["action"] + "</div>" if s["num"] == current["num"] else ""}
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
    # Extract key fields
    phase  = re.search(r"\*\*Phase:\*\*\s*(.+)", raw)
    status = re.search(r"\*\*Status:\*\*\s*(.+)", raw)
    nxt    = re.search(r"## Next action\n(.+)", raw)
    phase_txt  = phase.group(1).strip()  if phase  else "—"
    status_txt = status.group(1).strip() if status else "—"
    next_txt   = nxt.group(1).strip()   if nxt    else "—"
    color = "#3fb950" if "✓" in status_txt or "complete" in status_txt.lower() else \
            "#d29922" if "running" in status_txt.lower() else "#58a6ff"
    return f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px 20px;">
  <div style="font-size:12px;color:#8b949e;font-weight:500;letter-spacing:0.05em;margin-bottom:10px;">CURRENT STATE</div>
  <div style="margin-bottom:8px;">
    <span style="color:#8b949e;font-size:12px;">Phase</span><br>
    <span style="color:#e6edf3;font-size:14px;font-weight:600;">{phase_txt}</span>
  </div>
  <div style="margin-bottom:8px;">
    <span style="color:#8b949e;font-size:12px;">Status</span><br>
    <span style="color:{color};font-size:13px;">{status_txt}</span>
  </div>
  <div>
    <span style="color:#8b949e;font-size:12px;">Next action</span><br>
    <span style="color:#e6edf3;font-size:13px;font-family:monospace;">{next_txt}</span>
  </div>
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
        html = ""
        highlight = {"train_rows", "val_rows", "rows_raw", "license", "detected_format"}
        for k, v in card.items():
            col = "#58a6ff" if k in highlight else "#8b949e"
            html += f'<span style="color:{col};">{k}</span>: <span style="color:#e6edf3;">{v}</span><br>'
        return terminal_wrap(html, height=280)
    except Exception:
        return terminal_wrap(p.read_text())


def read_config_file(name: str) -> str:
    p = ROOT / "configs" / name
    return p.read_text() if p.exists() else "# Config not found"


def read_log(run_name: str) -> str:
    p = ROOT / "logs" / f"train_{run_name}.log"
    if not p.exists():
        return terminal_wrap('<span style="color:#d29922;">No log file yet</span>')
    lines = p.read_text().splitlines()[-60:]
    return terminal_wrap(ansi_to_html("\n".join(lines)), height=420)


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
    """Parse training log and return a live metrics panel."""
    if not run_name:
        return terminal_wrap("")
    p = ROOT / "logs" / f"train_{run_name}.log"
    if not p.exists():
        return terminal_wrap('<span style="color:#d29922;">No log file yet — start training first.</span>')

    lines = p.read_text().splitlines()

    # Parse latest step metrics
    step = grad = loss = lr = epoch = tok_s = None
    for line in reversed(lines):
        m = re.search(r"['\"]loss['\"]\s*[:=]\s*([\d.]+)", line)
        if m and loss is None:
            loss = float(m.group(1))
        m = re.search(r"['\"]grad_norm['\"]\s*[:=]\s*([\d.]+)", line)
        if m and grad is None:
            grad = float(m.group(1))
        m = re.search(r"['\"]step['\"]\s*[:=]\s*(\d+)", line)
        if m and step is None:
            step = int(m.group(1))
        m = re.search(r"['\"]learning_rate['\"]\s*[:=]\s*([\d.e-]+)", line)
        if m and lr is None:
            lr = float(m.group(1))
        m = re.search(r"['\"]epoch['\"]\s*[:=]\s*([\d.]+)", line)
        if m and epoch is None:
            epoch = float(m.group(1))
        if all(v is not None for v in [step, loss, grad, lr, epoch]):
            break

    if step is None:
        return terminal_wrap('<span style="color:#d29922;">Parsing log... no step data yet.</span>')

    # Estimate ETA from first and last timestamped steps
    eta_str = "—"
    timestamps = re.findall(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", p.read_text())
    if len(timestamps) >= 2:
        try:
            from datetime import datetime as _dt
            t0 = _dt.strptime(timestamps[0],  "%Y-%m-%d %H:%M:%S")
            t1 = _dt.strptime(timestamps[-1], "%Y-%m-%d %H:%M:%S")
            elapsed = (t1 - t0).total_seconds()
            if elapsed > 0 and step and step > 0:
                sps = step / elapsed  # steps per second
                # estimate total steps from log
                total_m = re.search(r"(\d+)/(\d+)\s*\[", "\n".join(lines[-20:]))
                if total_m:
                    total = int(total_m.group(2))
                    remaining = (total - step) / sps
                    h, m_ = int(remaining // 3600), int((remaining % 3600) // 60)
                    eta_str = f"~{h}h {m_}m"
        except Exception:
            pass

    # Color for loss
    loss_col = "#3fb950" if loss < 1.5 else ("#d29922" if loss < 3.0 else "#f85149")
    grad_col = "#3fb950" if grad < 2.0 else ("#d29922" if grad < 5.0 else "#f85149")

    def metric(label, value, color="#e6edf3", unit=""):
        return f"""
<div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;
            padding:10px 14px;flex:1;min-width:100px;">
  <div style="color:#8b949e;font-size:11px;text-transform:uppercase;
              letter-spacing:0.06em;margin-bottom:4px;">{label}</div>
  <div style="color:{color};font-size:20px;font-weight:700;font-family:monospace;">
    {value}<span style="font-size:12px;color:#8b949e;margin-left:3px;">{unit}</span>
  </div>
</div>"""

    html = f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;">
  <div style="font-size:12px;color:#8b949e;letter-spacing:0.05em;
              margin-bottom:12px;font-weight:500;">LIVE TRAINING METRICS — {run_name}</div>
  <div style="display:flex;gap:8px;flex-wrap:wrap;">
    {metric("Step",       step,                         "#58a6ff")}
    {metric("Loss",       f"{loss:.4f}" if loss else "—", loss_col)}
    {metric("Grad Norm",  f"{grad:.3f}" if grad else "—", grad_col)}
    {metric("Epoch",      f"{epoch:.2f}" if epoch else "—", "#e6edf3")}
    {metric("Learn Rate", f"{lr:.1e}" if lr else "—",   "#bc8cff")}
    {metric("ETA",        eta_str,                       "#d29922")}
  </div>
</div>"""
    return html


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


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
footer { display:none !important; }

/* Force all text inputs to be visible on dark background */
* { box-sizing: border-box; }

input:not([type=range]),
textarea {
  color: #e6edf3 !important;
  background-color: #161b22 !important;
  border-color: #30363d !important;
  caret-color: #e6edf3 !important;
}

input:not([type=range])::placeholder,
textarea::placeholder {
  color: #6e7681 !important;
  opacity: 1 !important;
}

/* Gradio 6 specific selectors */
.svelte-1f354aw,
[class*="svelte"] input,
[class*="svelte"] textarea,
.block input,
.block textarea,
.form input,
.form textarea {
  color: #e6edf3 !important;
  background-color: #161b22 !important;
}

/* Labels */
.block label span,
.block > label,
.form > label,
span.text-sm {
  color: #c9d1d9 !important;
}

/* Buttons */
button.primary {
  background: #238636 !important;
  border-color: #238636 !important;
  color: #fff !important;
  font-weight: 600 !important;
}
button.primary:hover { background: #2ea043 !important; }
button.secondary {
  background: #21262d !important;
  border-color: #30363d !important;
  color: #c9d1d9 !important;
}
button.secondary:hover { background: #30363d !important; }

/* Tabs */
.tab-nav button { color: #8b949e !important; background: transparent !important; }
.tab-nav button.selected { color: #e6edf3 !important; border-bottom-color: #58a6ff !important; }

/* Dropdown list options */
ul[role="listbox"], ul.options, .dropdown-menu,
[data-testid="dropdown-options"],
.svelte-1occ011, li[role="option"] {
  background: #161b22 !important;
  color: #e6edf3 !important;
}
li[role="option"]:hover, li[role="option"].selected {
  background: #21262d !important;
  color: #e6edf3 !important;
}
li[role="option"] * { color: #e6edf3 !important; }

/* Sliders */
input[type=range]::-webkit-slider-thumb { background: #58a6ff !important; }

/* Code */
.cm-content, .cm-line { color: #e6edf3 !important; }

/* Chat bubbles */
.message-bubble-border { border-color: #30363d !important; }
.bot, .bot * { background: #161b22 !important; color: #e6edf3 !important; }
.user, .user * { background: #1f6feb !important; color: #ffffff !important; }
[data-testid="bot"] { background: #161b22 !important; color: #e6edf3 !important; }
[data-testid="user"] { background: #1f6feb !important; color: #ffffff !important; }
.message { color: #e6edf3 !important; }
.message p, .message span, .message div { color: #e6edf3 !important; }
.prose p, .prose span { color: #e6edf3 !important; }
"""

# ── App ───────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MKD Fine-Tuning Platform") as app:

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
    # Guide panel
    with gr.Row():
        guide_html = gr.HTML(get_guide_html())

    refresh_all_btn.click(
        lambda: (get_pipeline_html(), get_stats_html(), get_state_html(),
                 get_gpu_html(), get_guide_html()),
        outputs=[pipeline_html, stats_html, state_html, gpu_html, guide_html],
    )

    # Auto-refresh guide too
    dash_timer.tick(
        lambda: (get_gpu_html(), get_pipeline_html(), get_stats_html(),
                 get_state_html(), get_guide_html()),
        outputs=[gpu_html, pipeline_html, stats_html, state_html, guide_html],
    )

    gr.HTML('<div style="border-top:1px solid #30363d;margin:8px 0 16px;"></div>')

    # ── Main tabs ─────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── 1 · Add Dataset ───────────────────────────────────────────────────
        with gr.Tab("1 · Dataset"):
            gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Add a dataset</h3>'
                    '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                    'Choose one source: HuggingFace / URL, or upload a file from your computer. '
                    'Auto-detects: Alpaca · ShareGPT · OpenAI · ChatML · DPO</p>')
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("HuggingFace / URL"):
                            ds_src = gr.Textbox(
                                label="HuggingFace dataset ID or URL",
                                placeholder="tatsu-lab/alpaca",
                            )
                            ds_name_hf = gr.Textbox(
                                label="Local name  (optional)",
                                placeholder="alpaca")
                            add_ds_hf_btn = gr.Button("Download & Normalize", variant="primary")

                        with gr.Tab("Upload from computer"):
                            ds_file = gr.File(
                                label="Select a file from your computer",
                                file_types=[".jsonl", ".json", ".csv", ".parquet"],
                                file_count="single",
                            )
                            ds_name_file = gr.Textbox(
                                label="Local name  (optional — defaults to filename)",
                                placeholder="leave blank to use filename")
                            add_ds_file_btn = gr.Button("Upload & Normalize", variant="primary")

                        with gr.Tab("Already on server"):
                            gr.HTML('<p style="color:#8b949e;font-size:12px;margin:8px 0;">'
                                    'Select one or multiple datasets. If you select more than one they will be merged.</p>')
                            existing_ds_cb = gr.CheckboxGroup(
                                label="Select datasets  (check one or more)",
                                choices=list_datasets(),
                            )
                            with gr.Row():
                                existing_ds_ref = gr.Button("↺ Refresh", size="sm", variant="secondary")
                                existing_ds_use = gr.Button("Use selected", variant="primary")
                            existing_ds_info = gr.HTML(terminal_wrap(""))

                with gr.Column(scale=3):
                    add_ds_out = gr.HTML(terminal_wrap(""))

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
                    return terminal_wrap('<span style="color:#f85149;">Select at least one dataset.</span>')
                if len(names) == 1:
                    cleaned = (ROOT / "datasets" / names[0] / "train.jsonl").exists()
                    next_tab = "4 · Configure" if cleaned else "2 · Clean"
                    return terminal_wrap(
                        f'<span style="color:#3fb950;font-weight:600;">Ready: {names[0]}</span><br>'
                        f'<span style="color:#8b949e;">Go to <strong>{next_tab}</strong>.</span>'
                    )
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
                    return terminal_wrap('<span style="color:#f85149;">No cleaned data found. Clean each dataset first.</span>')
                out = merged_dir / "normalized.jsonl"
                with out.open("w") as f:
                    for r in rows:
                        f.write(_j.dumps(r, ensure_ascii=False) + "\n")
                return terminal_wrap(
                    f'<span style="color:#3fb950;font-weight:600;">Merged {len(names)} datasets → "{merged_name}"</span><br>'
                    f'<span style="color:#8b949e;">{len(rows)} total rows. Go to <strong>2 · Clean</strong> to clean the merged dataset.</span>'
                ) + next_step_card("add-dataset")

            add_ds_hf_btn.click(do_add_hf, inputs=[ds_src, ds_name_hf],
                                outputs=add_ds_out, show_progress=False)
            add_ds_file_btn.click(do_upload, inputs=[ds_file, ds_name_file],
                                  outputs=add_ds_out, show_progress=False)
            existing_ds_ref.click(lambda: gr.CheckboxGroup(choices=list_datasets()), outputs=existing_ds_cb)
            existing_ds_cb.change(show_existing_ds, inputs=existing_ds_cb, outputs=existing_ds_info)
            existing_ds_use.click(use_existing_ds, inputs=existing_ds_cb, outputs=existing_ds_info)

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
            gr.HTML('<h4 style="color:#c9d1d9;margin:0 0 10px;">Sample preview & data card</h4>')
            with gr.Row():
                with gr.Column(scale=1):
                    sample_dd = gr.Dropdown(label="Dataset to preview",
                                             choices=list_cleaned_datasets(),
                                             allow_custom_value=True)
                    gr.Button("↺ Refresh", size="sm", variant="secondary").click(
                        lambda: gr.Dropdown(choices=list_cleaned_datasets()), outputs=sample_dd)
                with gr.Column(scale=2):
                    sample_out = gr.HTML(terminal_wrap(""))
                with gr.Column(scale=2):
                    card_out = gr.HTML(terminal_wrap(""))

            def do_clean(n, mc, vr, nopii):
                args = ["clean-dataset", n, "--max-chars", str(int(mc)), "--val-ratio", str(vr)]
                if nopii: args.append("--no-pii")
                yield from run_ft(args)

            clean_ref_btn.click(lambda: gr.Dropdown(choices=list_datasets()), outputs=clean_dd)
            clean_btn.click(
                do_clean,
                inputs=[clean_dd, clean_max, clean_val, clean_nopii],
                outputs=clean_out, show_progress=False,
            )
            sample_dd.change(read_sample,    inputs=sample_dd, outputs=sample_out)
            sample_dd.change(read_data_card, inputs=sample_dd, outputs=card_out)

        # ── 3 · Add Model ─────────────────────────────────────────────────────
        with gr.Tab("3 · Model"):
            gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Select a base model</h3>'
                    '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                    'Use a model already on the server, or download a new one.</p>')
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.Tab("Already on server"):
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
                    return terminal_wrap('<span style="color:#f85149;">Select a model first.</span>')
                return terminal_wrap(
                    f'<span style="color:#3fb950;font-weight:bold;">Model "{name}" selected.</span><br>'
                    f'<span style="color:#8b949e;">Go to <strong>4 · Configure</strong> and select this model.</span>'
                ) + next_step_card("add-model")

            def do_add_model(s, a):
                s = s.strip()
                yield from run_ft(["add-model", s] + (["--accept-license"] if a else []))

            model_server_ref.click(lambda: gr.Dropdown(choices=list_models()), outputs=existing_model_dd)
            existing_model_dd.change(show_model_info, inputs=existing_model_dd, outputs=model_server_info)
            model_server_use.click(use_server_model, inputs=existing_model_dd, outputs=model_server_info)
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

        # ── 5 · Train ─────────────────────────────────────────────────────────
        with gr.Tab("5 · Train"):
            gr.HTML('<h3 style="color:#e6edf3;margin:0 0 4px;">Smoke test → Full training</h3>'
                    '<p style="color:#8b949e;font-size:13px;margin:0 0 16px;">'
                    'Always run smoke test first · trains via Axolotl · live watchdog for NaN/OOM</p>')
            with gr.Row():
                train_cfg_dd = gr.Dropdown(label="Recipe", choices=list_configs(),
                                            allow_custom_value=True, scale=3)
                gr.Button("↺", size="sm", variant="secondary", scale=0).click(
                    lambda: gr.Dropdown(choices=list_configs()), outputs=train_cfg_dd)

            gr.HTML("""
            <div style="background:#161b22;border:1px solid #d29922;border-radius:8px;
                        padding:10px 16px;margin-bottom:12px;font-size:13px;color:#d29922;">
              !  Always run <strong>Smoke Test</strong> before full training.
              It confirms memory, loss shape, and checkpoint save/load — takes ~2 min.
            </div>
            """)
            smoke_steps = gr.Slider(
                minimum=10, maximum=200, value=50, step=10,
                label="Smoke test steps  (more steps = more confidence, longer wait)",
            )
            gr.HTML('<p style="color:#8b949e;font-size:12px;margin:-8px 0 12px;">'
                    '10 steps ≈ 30 sec · 50 steps ≈ 2 min · 200 steps ≈ 8 min</p>')
            with gr.Row():
                smoke_btn = gr.Button("Smoke Test", variant="secondary", scale=1)
                train_btn = gr.Button("Start Full Training", variant="primary", scale=1)

            train_out = gr.HTML(terminal_wrap(""))

            gr.HTML('<div style="border-top:1px solid #30363d;margin:16px 0;"></div>')
            gr.HTML('<h4 style="color:#c9d1d9;margin:0 0 10px;">Live training monitor</h4>')
            with gr.Row():
                log_dd  = gr.Dropdown(label="Run", choices=list_checkpoints(),
                                       allow_custom_value=True, scale=3)
                log_ref = gr.Button("↺ Refresh", size="sm", variant="secondary", scale=0)

            metrics_out = gr.HTML("")
            log_out = gr.HTML(terminal_wrap(""))

            # Auto-refresh metrics every 10s when training is active
            metrics_timer = gr.Timer(value=10)
            metrics_timer.tick(
                lambda r: (get_metrics_html(r), read_log(r)),
                inputs=log_dd, outputs=[metrics_out, log_out],
            )

            def do_smoke(c, steps):
                yield from run_ft(["smoke-test", f"configs/{c}",
                                   "--steps", str(int(steps))])

            def do_train(c):
                yield from run_ft(["train", f"configs/{c}"])

            smoke_btn.click(do_smoke, inputs=[train_cfg_dd, smoke_steps],
                            outputs=train_out, show_progress=False)
            train_btn.click(do_train, inputs=train_cfg_dd, outputs=train_out, show_progress=False)
            log_ref.click(lambda r: (get_metrics_html(r), read_log(r)),
                          inputs=log_dd, outputs=[metrics_out, log_out])
            log_dd.change(lambda r: (get_metrics_html(r), read_log(r)),
                          inputs=log_dd, outputs=[metrics_out, log_out])

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
                    def do_publish(exp, hf, gh, ack, lic):
                        args = ["publish", f"exports/{exp}", "--hf-repo", hf]
                        if gh.strip(): args += ["--github-repo", gh]
                        if ack: args.append("--acknowledge-contamination")
                        if lic: args.append("--accept-base-license")
                        yield from run_ft(args)

                    pub_btn.click(do_publish, inputs=[pub_dd, hf_repo, gh_repo, pub_ack, pub_lic],
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
                        str(model_path), trust_remote_code=False, local_files_only=True)
                    _chat_model = AutoModelForCausalLM.from_pretrained(
                        str(model_path), device_map="auto",
                        torch_dtype=torch.bfloat16, trust_remote_code=False,
                        local_files_only=True,
                    )
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
                    ids = _chat_tok(text, return_tensors="pt").input_ids.to(_chat_model.device)
                    with torch.no_grad():
                        out = _chat_model.generate(
                            ids, max_new_tokens=max_tokens,
                            do_sample=True, temperature=0.7, top_p=0.9,
                            pad_token_id=_chat_tok.eos_token_id,
                        )
                    reply = _chat_tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
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

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=theme,
        css=CSS,
    )
