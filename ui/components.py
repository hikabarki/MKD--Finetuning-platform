"""
ui/components.py — UI helper functions: ANSI renderer, data listers, HTML panels, ft runner.
"""
from __future__ import annotations

import json
import ast
import os
import re
import subprocess

from inference.state import ROOT, SCRIPTS, PYTHON

# ── ANSI → HTML terminal renderer ─────────────────────────────────────────────
_ANSI_COLOR = {
    "32": "#3fb950", "33": "#d29922", "31": "#f85149",
    "36": "#39c5cf", "35": "#bc8cff", "34": "#58a6ff",
    "2":  "#8b949e", "1":  None,
}


def ansi_to_html(text: str) -> str:
    text      = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    open_span = False

    def replace(m):
        nonlocal open_span
        codes = m.group(1).split(";")
        if codes == ["0"] or codes == [""]:
            end       = "</span>" if open_span else ""
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
    return html


def terminal_wrap(content: str, height: int = 380) -> str:
    return f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;
            font-family:'JetBrains Mono',Consolas,monospace;font-size:13px;
            color:#e6edf3;line-height:1.65;padding:16px;
            max-height:{height}px;overflow-y:auto;white-space:pre-wrap;">
{content if content else '<span style="color:#8b949e">Output will appear here...</span>'}
</div>"""


# ── Data listers ───────────────────────────────────────────────────────────────
def list_datasets() -> list[str]:
    d = ROOT / "datasets"
    return [p.name for p in d.iterdir() if p.is_dir()] if d.exists() else []


def list_cleaned_datasets() -> list[str]:
    d = ROOT / "datasets"
    return [p.name for p in d.iterdir()
            if p.is_dir() and (p / "train.jsonl").exists()] if d.exists() else []


def list_models() -> list[str]:
    d = ROOT / "models"
    return [p.name for p in d.iterdir() if p.is_dir()] if d.exists() else []


def list_configs() -> list[str]:
    d = ROOT / "configs"
    return sorted([p.name for p in d.glob("run_*.yaml")], reverse=True) if d.exists() else []


def list_checkpoints() -> list[str]:
    d = ROOT / "checkpoints"
    return sorted([p.name for p in d.iterdir() if p.is_dir()], reverse=True) if d.exists() else []


def list_exports() -> list[str]:
    d = ROOT / "exports"
    return sorted([p.name for p in d.iterdir() if p.is_dir()], reverse=True) if d.exists() else []


# ── GPU / pipeline / stats panels ─────────────────────────────────────────────
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
            util_i  = int(util)
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
      <div style="width:{util_i}%;background:{bar_col};border-radius:4px;height:8px;transition:width 0.3s;"></div>
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
    steps = [
        {"num": 1, "label": "Add Dataset",    "done": bool(list_datasets()),
         "tab": "1 · Dataset",              "action": 'Open <b>1 · Dataset</b> and download a dataset.'},
        {"num": 2, "label": "Clean Dataset",  "done": bool(list_cleaned_datasets()),
         "tab": "2 · Clean",                "action": 'Open <b>2 · Clean</b> and clean your dataset.'},
        {"num": 3, "label": "Add Model",      "done": bool(list_models()),
         "tab": "3 · Model",                "action": 'Open <b>3 · Model</b> and download a base model.'},
        {"num": 4, "label": "Configure",      "done": bool(list_configs()),
         "tab": "4 · Configure",            "action": 'Open <b>4 · Configure</b> and generate a recipe.'},
        {"num": 5, "label": "Train",
         "done": any(any((ROOT/"checkpoints"/d).glob("checkpoint-*")) for d in list_checkpoints()) if list_checkpoints() else False,
         "tab": "5 · Train",                "action": 'Open <b>5 · Train</b> — run Smoke Test, then Start Training.'},
        {"num": 6, "label": "Evaluate",
         "done": any((ROOT/"eval_results"/d/"summary.md").exists() for d in list_checkpoints()) if list_checkpoints() else False,
         "tab": "6 · Eval",                 "action": 'Open <b>6 · Eval</b> and run evaluation.'},
        {"num": 7, "label": "Export",
         "done": bool(list_exports()) and any(
             any((ROOT/"exports"/d/"merged").glob("model*.safetensors")) for d in list_exports()
         ) if list_exports() else False,
         "tab": "7 · Export & Publish",     "action": 'Open <b>7 · Export &amp; Publish</b> and export your model.'},
        {"num": 8, "label": "Publish",       "done": False,
         "tab": "7 · Export & Publish",     "action": 'Open <b>7 · Export &amp; Publish</b> and publish to HuggingFace.'},
    ]

    current = next((s for s in steps if not s["done"]), steps[-1])
    rows    = ""
    for s in steps:
        if s["done"]:
            icon       = '<span style="color:#3fb950;font-weight:700;">✓</span>'
            color      = "#3fb950"
            text_color = "#8b949e"
            bg         = "transparent"
        elif s["num"] == current["num"]:
            icon       = f'<span style="background:#1f6feb;color:#fff;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">{s["num"]}</span>'
            color      = "#58a6ff"
            text_color = "#e6edf3"
            bg         = "#0d1f3c"
        else:
            icon       = f'<span style="background:#21262d;color:#8b949e;border-radius:50%;width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;">{s["num"]}</span>'
            color      = "#484f58"
            text_color = "#484f58"
            bg         = "transparent"

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
    raw    = p.read_text()
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


# ── Next step card ─────────────────────────────────────────────────────────────
_NEXT_TAB = {
    "add-dataset":   ("2 · Clean",          "Go to the Clean tab to deduplicate and split your dataset."),
    "clean-dataset": ("3 · Model",           "Go to the Model tab to download a base model."),
    "add-model":     ("4 · Configure",       "Go to Configure to generate your training recipe."),
    "configure":     ("5 · Train",           "Go to Train — run Smoke Test first, then full training."),
    "smoke-test":    ("5 · Train",           "Smoke test passed. Click Start Full Training."),
    "train":         ("6 · Eval",            "Training complete. Go to Eval to benchmark your model."),
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
    cmd  = [PYTHON, str(SCRIPTS / "ft")] + args
    env  = {**os.environ, "FT_PROJECT": str(ROOT),
            "PYTHONUNBUFFERED": "1", "WANDB_DISABLED": "true", "FT_UI": "1"}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, cwd=str(ROOT))
    output = ""
    for line in iter(proc.stdout.readline, ""):
        output += line
        yield terminal_wrap(ansi_to_html(output))
    proc.wait()
    rc      = proc.returncode
    command = args[0] if args else ""
    if rc == 0:
        tail = '\n<span style="color:#3fb950;font-weight:bold;">✓ Done</span>'
        card = next_step_card(command)
        yield terminal_wrap(ansi_to_html(output) + tail) + card
    else:
        tail = f'\n<span style="color:#f85149;font-weight:bold;">✗ Failed (exit {rc})</span>'
        yield terminal_wrap(ansi_to_html(output) + tail)


# ── File readers ───────────────────────────────────────────────────────────────
def read_sample(name: str) -> str:
    p = ROOT / "datasets" / name / "sample_100.jsonl"
    if not p.exists():
        return terminal_wrap('<span style="color:#d29922;">No sample — run clean-dataset first</span>')
    lines = [json.loads(l) for l in p.read_text().splitlines()[:4] if l.strip()]
    html  = ""
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
            val_str = str(val)
            if "fill in before" in val_str.lower():
                val_str = "UNKNOWN"
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
        return (
            f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
            f'padding:14px 18px;">'
            f'<div style="color:#8b949e;font-size:10px;font-weight:600;letter-spacing:0.06em;'
            f'margin-bottom:10px;">DATA CARD</div>'
            f'{rows_html}'
            f'</div>'
        )
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
        if re.match(r"^\s*\d+%\|", line):
            continue
        if not line or set(line) <= {"-", "=", " "}:
            continue
        if re.match(r"^\s{2,}\"[a-z_]+\":", line):
            continue
        if any(x in line for x in (
            "'loss'", "'eval_loss'", "ERROR", "WARNING", "ABORT",
            "✓", "✗", "Training complete", "Evaluating", "Saving",
            "epoch", "steps_per_second",
        )):
            important.append(line)

    if not important:
        return terminal_wrap('<span style="color:#8b949e;">No log events yet.</span>', height=220)

    shown   = important[-25:]
    colored = []
    for line in shown:
        if "'loss'" in line and "'eval_loss'" not in line:
            try:
                d       = ast.literal_eval(line)
                step_n  = len([l for l in important if "'loss'" in l and "'eval_loss'" not in l])
                loss_v  = float(d.get("loss", 0))
                ppl_v   = float(d.get("ppl", 0))
                epoch_v = float(d.get("epoch", 0))
                lc      = "32" if loss_v < 1.5 else ("33" if loss_v < 3.0 else "31")
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
                d  = ast.literal_eval(line)
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
