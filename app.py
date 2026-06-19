#!/usr/bin/env python3
"""
app.py — MKD Fine-Tuning Platform entry point.

Launch:  ft ui  |  python app.py  |  python app.py --port 7860 --share
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import ast

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── Module imports (extracted in Phase 3) ─────────────────────────────────────
from inference import state as _state
from inference.state import (
    ROOT, SCRIPTS, PYTHON, VLLM_PYTHON, VLLM_PORT,
    DEFAULT_BASE_MODEL, DEFAULT_BASE_DIR, _FALLBACK_BASE_DIR,
    NEMOTRON_HF_ID, _MODEL_DISPLAY_INFO,
    _chat_lock, _vllm_lock, _user_model_path_lock, _user_training_lock,
)
from inference.vllm_server import _is_vllm_model, _start_vllm_server, _stop_vllm_server
from inference.chat import (
    _is_vision_model, _is_vl_model, _nvfp4_quant_config,
    _load_chat_model, _build_chat_inputs, _generate_reply,
)
from ui.components import (
    ansi_to_html, terminal_wrap,
    list_datasets, list_cleaned_datasets, list_models, list_configs,
    list_checkpoints, list_exports,
    get_gpu_html, get_pipeline_html, get_stats_html, get_guide_html,
    get_state_html, next_step_card, run_ft,
    read_sample, read_data_card, read_config_file, read_log,
    read_eval_summary, read_model_card, save_model_card,
)
from ft_platform.log_parser import (
    list_eval_runs, get_loss_df, get_metrics_html, parse_training_log,
)
from ft_platform.eval_dashboard import (
    build_loss_chart, build_ppl_chart, build_summary_chart,
    build_metrics_df, load_eval_run,
)
from ui.styles import CSS


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
                    minimum=1,
                    info="Number of steps for the quick test. More steps = better evaluation metrics.",
                    visible=True,
                )
                user_qs_max_steps = gr.Number(
                    label="Max training steps",
                    value=0,
                    precision=0,
                    minimum=0,
                    info="Cap the full training run (0 = no limit, runs full epochs).",
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
            # VL 12B first so it appears as the top radio option
            preferred_order = [DEFAULT_BASE_DIR, _FALLBACK_BASE_DIR]
            all_models = list_models()
            ordered = [m for m in preferred_order if m in all_models] + \
                      [m for m in all_models if m not in preferred_order]
            for m in ordered:
                # Exclude NVFP4/vLLM-only models — they cannot be fine-tuned
                if _is_vllm_model(str(ROOT / "models" / m)):
                    continue
                friendly = _MODEL_DISPLAY_INFO.get(m)
                # Format: "folder-name  (friendly info)" — folder-name prefix is used
                # by run_user_custommix for fuzzy-matching back to the folder.
                choices.append(f"{m}  ({friendly})" if friendly else m)
            return choices or [f"{_FALLBACK_BASE_DIR}  (Nemotron 4B · BF16  (4.3 GB VRAM))"]

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
                def _default_model_choice():
                    choices = get_model_choices()
                    # Prefer VL 12B, then any Nemotron, then first available
                    for priority in (DEFAULT_BASE_DIR, _FALLBACK_BASE_DIR):
                        for c in choices:
                            if c.startswith(priority):
                                return c
                    return choices[0] if choices else None

                user_cm_model = gr.Radio(
                    choices=get_model_choices(),
                    value=_default_model_choice(),
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

            def _auto_model_name(model_choice):
                import datetime, re as _re
                if not model_choice:
                    return ""
                # Take the folder-name part before "  (" and strip trailing spaces
                slug = model_choice.split("  (")[0].strip().lower()
                # Keep only alphanumeric and hyphens; collapse repeated dashes
                slug = _re.sub(r"[^a-z0-9]+", "-", slug)
                slug = _re.sub(r"-{2,}", "-", slug).strip("-")
                date = datetime.date.today().strftime("%b%d").lower()
                return f"{slug}-{date}"

            user_cm_name = gr.Textbox(
                label="What should we call your model?  (auto-filled — you can change it)",
                value=_auto_model_name(_default_model_choice()),
                placeholder="e.g.  my-assistant  or  support-bot",
            )
            user_cm_model.change(_auto_model_name, inputs=user_cm_model, outputs=user_cm_name)

            gr.HTML('<div style="height:8px;"></div>')
            user_cm_epochs = gr.Radio(
                choices=["Fast — 1 epoch (~1-3 hrs)", "Standard — 3 epochs (~6-9 hrs)"],
                value="Fast — 1 epoch (~1-3 hrs)",
                label="Training depth",
                info="Fast is good for testing. Standard gives better results.",
            )

            gr.HTML('<div style="height:8px;"></div>')
            user_cm_max_steps = gr.Number(
                value=0,
                minimum=0,
                label="Max training steps  (0 = run full epochs above)",
                info="0 = complete all epochs · 100 steps ≈ 5 min · 500 ≈ 30 min · 1000 ≈ 1 hr  |  Useful to cap a long run early.",
            )

            # Hidden — always 50, no need to expose to general users
            user_cm_test_steps = gr.Number(value=50, visible=False)

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
            user_chatbot = gr.Chatbot(height=360, label="Chat with your fine-tuned model")
            with gr.Accordion("📷 Upload image (optional — for vision/multimodal models)", open=False):
                user_chat_image = gr.Image(type="pil", label="Image", height=200)
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
            with gr.Accordion("🔍 Error Log", open=False):
                user_error_log = gr.Textbox(
                    label="Recent errors (most recent first)",
                    value="No errors yet.",
                    lines=8,
                    max_lines=20,
                    interactive=False,
                    elem_id="user_error_log",
                )
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

    # All group outputs: landing, dev_group, user_group, dev_content, eval_group
    _outs = [landing_group, dev_group, user_group, dev_content_group, eval_group]

    def _g(v): return gr.Group(visible=v)

    def go_to_dev():
        return _g(False), _g(True), _g(False), _g(True), _g(False)

    def go_user():
        return _g(False), _g(False), _g(True), _g(False), _g(False)

    def go_home():
        return _g(True), _g(False), _g(False), _g(False), _g(False)

    def go_eval():
        runs = list_eval_runs()
        default = (runs or [None])[0]
        loss_fig, ppl_fig, summary_fig, df = load_eval_run(default)
        has_data = bool(default)
        return (
            _g(False), _g(False), _g(False), _g(False), _g(True),
            gr.Dropdown(choices=runs, value=default),
            loss_fig, ppl_fig, summary_fig, df,
            gr.HTML(visible=not has_data),
            gr.Group(visible=has_data),
        )

    land_dev_btn.click(go_to_dev,  outputs=_outs)
    land_user_btn.click(go_user,   outputs=_outs)
    land_eval_btn.click(go_eval,   outputs=_outs + [
        eval_run_dd, eval_loss_plot, eval_ppl_plot, eval_summary_plot,
        eval_table, eval_no_data, eval_charts_group,
    ])

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

    def run_user_quickstart(dataset_choice, upload_file, model_name, test_mode=True, test_steps=50, max_steps=0):
        dataset = "alpaca"
        # Priority: VL 12B → Nemotron 4B BF16 → Qwen 7B → Qwen 3B
        _model_priority = [
            DEFAULT_BASE_DIR,   # nvidia__nvidia-nemotron-nano-12b-v2-vl-bf16  (VL 12B — default)
            _FALLBACK_BASE_DIR, # nvidia__nvidia-nemotron-3-nano-4b-bf16        (4B fallback)
        ]
        model = DEFAULT_BASE_DIR
        for _m in _model_priority:
            if (ROOT / "models" / _m).exists():
                model = _m
                break
        test_steps = int(test_steps or 50)
        max_steps  = int(max_steps  or 0)

        # If customer uploaded their own file, prepare it first (add → clean)
        if dataset_choice != "Use the built-in example dataset" and upload_file is not None:
            import re as _re_qs
            _upath = Path(upload_file.name if hasattr(upload_file, "name") else str(upload_file))
            _ds_slug = _re_qs.sub(r"[^a-z0-9]+", "-", _upath.stem.lower()).strip("-")[:30] or "my-dataset"
            yield build_progress_html("prepare"), terminal_wrap(f"Preparing your data: {_upath.name}…")
            for chunk in run_ft(["add-dataset", str(_upath), "--name", _ds_slug]):
                yield build_progress_html("prepare"), chunk
            for chunk in run_ft(["clean-dataset", _ds_slug]):
                yield build_progress_html("prepare"), chunk
            dataset = _ds_slug
            test_mode = False   # real data — skip the alpaca_tiny shortcut

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

        import subprocess as _sp_qs
        model_path = str(ROOT / "models" / model)
        is_vl = _is_vl_model(model_path)

        # ── VL path ──────────────────────────────────────────────────────────
        if is_vl:
            ds_path = str(ROOT / "datasets" / dataset)
            if not Path(ds_path).exists():
                ds_path = str(ROOT / "cleaned_datasets" / dataset)
            out_path = str(ROOT / "checkpoints" / f"vl_{model}")
            epochs_vl = 1
            cmd = [
                "python", str(ROOT / "vl_trainer.py"),
                "--model", model_path,
                "--dataset", ds_path,
                "--output", out_path,
                "--quant", "bf16",
                "--lora-r", "16",
                "--lr", "2e-4",
                "--epochs", str(epochs_vl),
            ]
            if test_mode:
                cmd += ["--max-steps", str(test_steps)]
            elif max_steps > 0:
                cmd += ["--max-steps", str(max_steps)]

            yield build_progress_html("configure"), terminal_wrap("Setting up VL training...")
            yield build_progress_html("smoke"), terminal_wrap("Starting VL trainer (smoke + train + merge)...")

            env = {**os.environ,
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "HF_HUB_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false"}
            export_path = None
            vqa_metrics = {}
            buf = []
            with _sp_qs.Popen(cmd, stdout=_sp_qs.PIPE, stderr=_sp_qs.STDOUT, text=True, env=env) as proc:
                for line in proc.stdout:
                    line = line.rstrip()
                    buf.append(line)
                    if line.startswith("EXPORT_PATH="):
                        export_path = line.split("=", 1)[1].strip()
                    elif line.startswith("VQA_METRICS="):
                        try:
                            vqa_metrics = json.loads(line.split("=", 1)[1])
                        except Exception:
                            pass
                    step = "train" if "step" in line.lower() or "loss" in line.lower() else \
                           "export" if "merging" in line.lower() or "export" in line.lower() else "smoke"
                    yield build_progress_html(step), terminal_wrap("\n".join(buf[-150:]))

            if export_path and Path(export_path).exists():
                _user_last_model_path["path"] = export_path
            else:
                exp_dir = ROOT / "exports"
                if exp_dir.exists():
                    candidates = sorted(
                        [d / "merged" for d in exp_dir.iterdir() if (d / "merged" / "config.json").exists()],
                        key=lambda p: p.stat().st_mtime, reverse=True)
                    if candidates:
                        _user_last_model_path["path"] = str(candidates[0])

            # Register VQA metrics in eval dashboard (same as Custom Mix VL path)
            if vqa_metrics:
                import datetime as _dt_qs
                _qs_vl_run = f"vl_{model}_{_dt_qs.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                _qs_log_dir = ROOT / "logs"; _qs_log_dir.mkdir(exist_ok=True)
                _qs_eval_dir = ROOT / "eval_results" / _qs_vl_run; _qs_eval_dir.mkdir(parents=True, exist_ok=True)
                _qs_log_lines = [
                    json.dumps({"loss": vqa_metrics.get("vqa_accuracy_before", 0), "ppl": 1.0}),
                    json.dumps({"loss": vqa_metrics.get("vqa_accuracy_after",  0), "ppl": 1.0}),
                    json.dumps({"vqa_accuracy_before": vqa_metrics["vqa_accuracy_before"],
                                "vqa_accuracy_after":  vqa_metrics["vqa_accuracy_after"],
                                "vqa_n_samples":       vqa_metrics.get("vqa_n_samples", 0)}),
                ]
                (_qs_log_dir / f"train_{_qs_vl_run}.log").write_text("\n".join(_qs_log_lines))
                (_qs_eval_dir / "summary.md").write_text(
                    f"# VL Training: {model}\n\n"
                    f"| Metric | Before | After | Change |\n|---|---|---|---|\n"
                    f"| VQA Accuracy | {vqa_metrics['vqa_accuracy_before']:.1%} | "
                    f"{vqa_metrics['vqa_accuracy_after']:.1%} | "
                    f"{(vqa_metrics['vqa_accuracy_after']-vqa_metrics['vqa_accuracy_before'])*100:+.1f}pp |\n"
                )

            yield build_progress_html("done"), terminal_wrap(
                'Waiting for GPU memory to clear before loading chat model...')
            _sp_qs.run(["pkill", "-9", "-f", "vl_trainer.py"], capture_output=True)
            _stop_vllm_server()
            import time as _time_qs
            import torch as _torch_qs
            for _wi in range(30):
                _torch_qs.cuda.empty_cache()
                if _torch_qs.cuda.mem_get_info()[0] / 1024 ** 3 >= 25.0:
                    break
                _time_qs.sleep(1)

            yield build_progress_html("done"), terminal_wrap(
                '<span style="color:#3fb950;font-weight:bold;font-size:16px;">Complete!</span><br>'
                '<span style="color:#8b949e;">Your VL model is ready. Click "Test in Chat" to try it.</span>'
            )
            return

        # ── Text (Axolotl) path ───────────────────────────────────────────────
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

        # Remove smoke-test cap; optionally inject user's max_steps for full training
        if test_mode or max_steps > 0:
            cfg_path = ROOT / "configs" / configs[0]
            if cfg_path.exists():
                import re as _re2
                txt = cfg_path.read_text()
                txt = _re2.sub(r'^max_steps:.*\n?', '', txt, flags=_re2.MULTILINE)
                if max_steps > 0:
                    txt = _re2.sub(r'^num_epochs:', f'max_steps: {max_steps}\nnum_epochs:',
                                   txt, flags=_re2.MULTILINE)
                cfg_path.write_text(txt)

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

            # Register run in eval_results so the Evaluation Dashboard can discover it
            _qs_run = new_ckpts[0]
            _qs_eval_dir = ROOT / "eval_results" / _qs_run
            _qs_eval_dir.mkdir(parents=True, exist_ok=True)
            if not (_qs_eval_dir / "summary.md").exists():
                (_qs_eval_dir / "summary.md").write_text(
                    f"# Training run: {_qs_run}\n\n"
                    f"Model: {model}  |  Dataset: {dataset}  |  Mode: Quick Start\n\n"
                    f"See the Training Loss and Perplexity charts for metrics.\n"
                )

        yield build_progress_html("done"), terminal_wrap(
            '<span style="color:#3fb950;font-weight:bold;font-size:16px;">Complete!</span>')


    def run_user_custommix(models_sel, datasets_sel, upload_file, model_name, test_steps=50, epochs="Fast — 1 epoch (~1-3 hrs)", max_steps=0):
        import subprocess as _sp
        test_steps = int(test_steps or 50)
        max_steps = int(max_steps or 0)
        if isinstance(epochs, str):
            epochs = 3 if "3" in epochs else 1
        else:
            epochs = int(epochs or 1)

        # Resolve model folder name from display label
        available = list_models()
        model = available[0] if available else ""
        label = (models_sel or "").split("  ")[0].strip().lower().replace("-", "").replace(".", "")
        for m in available:
            if label in m.lower().replace("-", "").replace(".", "").replace("_", ""):
                model = m
                break

        # Resolve dataset — handle customer file upload first
        ds_names = []
        _wants_upload = any("Upload" in (x or "") for x in (datasets_sel or []))
        if _wants_upload and upload_file is not None:
            import re as _re_cm
            _upath = Path(upload_file.name if hasattr(upload_file, "name") else str(upload_file))
            _ds_slug = _re_cm.sub(r"[^a-z0-9]+", "-", _upath.stem.lower()).strip("-")[:30] or "my-dataset"
            yield build_progress_html("prepare"), terminal_wrap(f"Preparing your data: {_upath.name}…")
            for chunk in run_ft(["add-dataset", str(_upath), "--name", _ds_slug]):
                yield build_progress_html("prepare"), chunk
            for chunk in run_ft(["clean-dataset", _ds_slug]):
                yield build_progress_html("prepare"), chunk
            ds_names = [_ds_slug]
        else:
            for d in (datasets_sel or []):
                ds_name = d.split("  ")[0].strip()
                if ds_name and "Upload" not in ds_name:
                    ds_names.append(ds_name)

        if not ds_names:
            available_ds = list_cleaned_datasets()
            ds_names = [available_ds[0]] if available_ds else ["alpaca"]
        dataset = ds_names[0]

        model_path = str(ROOT / "models" / model)
        is_vl = _is_vl_model(model_path)

        # ── VL path ──────────────────────────────────────────────────────────
        if is_vl:
            # M-5: Canonical dataset path is datasets/<name>/train.jsonl
            ds_path = str(ROOT / "datasets" / dataset)
            if not Path(ds_path).exists():
                ds_path = str(ROOT / "cleaned_datasets" / dataset)  # legacy fallback
            out_path = str(ROOT / "checkpoints" / f"vl_{model}")
            cmd = [
                "python", str(ROOT / "vl_trainer.py"),
                "--model", model_path,
                "--dataset", ds_path,
                "--output", out_path,
                "--quant", "bf16",
                "--lora-r", "16",
                "--lr", "2e-4",
                "--epochs", str(epochs),
            ]
            if max_steps > 0:
                cmd += ["--max-steps", str(max_steps)]

            yield build_progress_html("configure"), terminal_wrap("Setting up VL training...")
            yield build_progress_html("smoke"), terminal_wrap("Starting VL trainer (smoke + train + merge)...")

            env = {**os.environ,
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   "HF_HUB_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false"}
            export_path = None
            vqa_metrics = {}
            buf = []
            with _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, env=env) as proc:
                for line in proc.stdout:
                    line = line.rstrip()
                    buf.append(line)
                    if line.startswith("EXPORT_PATH="):
                        export_path = line.split("=", 1)[1].strip()
                    elif line.startswith("VQA_METRICS="):
                        try:
                            vqa_metrics = json.loads(line.split("=", 1)[1])
                        except Exception:
                            pass
                    step = "train" if "step" in line.lower() or "loss" in line.lower() else \
                           "export" if "merging" in line.lower() or "export" in line.lower() else "smoke"
                    yield build_progress_html(step), terminal_wrap("\n".join(buf[-150:]))

            if export_path and Path(export_path).exists():
                _user_last_model_path["path"] = export_path
            else:
                # Fallback: find newest export
                exp_dir = ROOT / "exports"
                if exp_dir.exists():
                    candidates = sorted(
                        [d / "merged" for d in exp_dir.iterdir() if (d / "merged" / "config.json").exists()],
                        key=lambda p: p.stat().st_mtime, reverse=True)
                    if candidates:
                        _user_last_model_path["path"] = str(candidates[0])

            # Write VQA metrics to log so the eval dashboard can find this run
            if vqa_metrics:
                import datetime as _dt_vqa
                _vl_run = f"vl_{model}_{_dt_vqa.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                _log_dir = ROOT / "logs"; _log_dir.mkdir(exist_ok=True)
                _eval_dir = ROOT / "eval_results" / _vl_run; _eval_dir.mkdir(parents=True, exist_ok=True)
                _log_lines = [
                    json.dumps({"loss": vqa_metrics.get("vqa_accuracy_before", 0), "ppl": 1.0}),
                    json.dumps({"loss": vqa_metrics.get("vqa_accuracy_after",  0), "ppl": 1.0}),
                    json.dumps({"vqa_accuracy_before": vqa_metrics["vqa_accuracy_before"],
                                "vqa_accuracy_after":  vqa_metrics["vqa_accuracy_after"],
                                "vqa_n_samples":       vqa_metrics.get("vqa_n_samples", 0)}),
                ]
                (_log_dir / f"train_{_vl_run}.log").write_text("\n".join(_log_lines))
                (_eval_dir / "summary.md").write_text(
                    f"# VL Training: {model}\n\n"
                    f"| Metric | Before | After | Change |\n|---|---|---|---|\n"
                    f"| VQA Accuracy | {vqa_metrics['vqa_accuracy_before']:.1%} | "
                    f"{vqa_metrics['vqa_accuracy_after']:.1%} | "
                    f"{(vqa_metrics['vqa_accuracy_after']-vqa_metrics['vqa_accuracy_before'])*100:+.1f}pp |\n"
                )

            # Kill ALL processes that hold VRAM and wait for memory to be fully released
            yield build_progress_html("done"), terminal_wrap(
                'Waiting for GPU memory to clear before loading chat model...')
            _sp.run(["pkill", "-9", "-f", "vl_trainer.py"], capture_output=True)
            _stop_vllm_server()  # kills vLLM server + orphaned EngineCore/VllmWorker
            import time as _time
            import torch as _torch_gc
            for _wi in range(30):
                _torch_gc.cuda.empty_cache()
                _free_gb = _torch_gc.cuda.mem_get_info()[0] / 1024 ** 3
                if _free_gb >= 25.0:
                    break
                _time.sleep(1)

            yield build_progress_html("done"), terminal_wrap(
                '<span style="color:#3fb950;font-weight:bold;font-size:16px;">Fine-tuning complete!</span><br>'
                '<span style="color:#8b949e;">Your VL model is ready. Click "Test in Chat" to try it.</span>'
            )
            return

        # ── Text (Axolotl) path ───────────────────────────────────────────────
        yield build_progress_html("configure"), terminal_wrap("Generating recipe...")
        cfg_args = ["configure", f"--model={model}", f"--dataset={dataset}",
                    "--quantize=4bit", f"--epochs={epochs}", "--lr=2e-4"]
        for chunk in run_ft(cfg_args):
            yield build_progress_html("configure"), chunk

        configs = list_configs()
        if not configs:
            yield build_progress_html("configure"), terminal_wrap('<span style="color:#f85149;">Config failed.</span>')
            return

        latest_cfg = f"configs/{configs[0]}"
        yield build_progress_html("smoke"), terminal_wrap("Running smoke test...")
        for chunk in run_ft(["smoke-test", latest_cfg, f"--steps={test_steps}"]):
            yield build_progress_html("smoke"), chunk

        yield build_progress_html("train"), terminal_wrap("Training started...")
        train_cmd = ["train", latest_cfg]
        if max_steps > 0:
            train_cmd += ["--max-steps", str(max_steps)]
        for chunk in run_ft(train_cmd):
            yield build_progress_html("train"), chunk

        new_ckpts = list_checkpoints()
        if new_ckpts:
            yield build_progress_html("export"), terminal_wrap("Exporting...")
            for chunk in run_ft(["export", f"checkpoints/{new_ckpts[0]}", "--merge", "--quantize=none"]):
                yield build_progress_html("export"), chunk
            exports = list_exports()
            if exports:
                _user_last_model_path["path"] = str(ROOT / "exports" / exports[0] / "merged")

            # Register run in eval_results so the Evaluation Dashboard can discover it
            _cm_run = new_ckpts[0]
            _cm_eval_dir = ROOT / "eval_results" / _cm_run
            _cm_eval_dir.mkdir(parents=True, exist_ok=True)
            if not (_cm_eval_dir / "summary.md").exists():
                (_cm_eval_dir / "summary.md").write_text(
                    f"# Training run: {_cm_run}\n\n"
                    f"Model: {model}  |  Dataset: {dataset}  |  Mode: Custom Mix\n\n"
                    f"See the Training Loss and Perplexity charts for metrics.\n"
                )

        yield build_progress_html("done"), terminal_wrap(
            '<span style="color:#3fb950;font-weight:bold;font-size:16px;">Fine-tuning complete!</span><br>'
            '<span style="color:#8b949e;">Your model is ready. Click "Test in Chat" to try it.</span>'
        )

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
        try:
            # Prefer: (1) just-trained VL export, (2) any VL export, (3) base VL model, (4) any export
            def _is_valid(p):
                mp = Path(p) if p else None
                return mp and mp.exists() and (mp / "tokenizer.json").exists() and (mp / "config.json").exists()

            model_path = _user_last_model_path.get("path")

            # If last trained model isn't VL-capable, search for a better one
            if not model_path or not _is_valid(model_path) or not _is_vl_model(model_path):
                # Look for a VL export first (newest first)
                vl_export = None
                exp_dir = ROOT / "exports"
                if exp_dir.exists():
                    for d in sorted(exp_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                        candidate = d / "merged"
                        if _is_valid(str(candidate)) and _is_vl_model(str(candidate)):
                            vl_export = str(candidate)
                            break
                if vl_export:
                    model_path = vl_export
                else:
                    # Fall back to base VL model from models/
                    models_dir = ROOT / "models"
                    if models_dir.exists():
                        for d in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                            if _is_valid(str(d)) and _is_vl_model(str(d)):
                                model_path = str(d)
                                break
                    # Last resort: any valid model
                    if not model_path or not _is_valid(model_path):
                        if models_dir.exists():
                            for d in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                                if _is_valid(str(d)):
                                    model_path = str(d)
                                    break

            if model_path and _is_valid(model_path):
                _user_last_model_path["path"] = model_path
        except Exception as _gtc_exc:
            import traceback as _gtc_tb
            with open("logs/ft_chat_errors.log", "a") as _ef:
                _ef.write(f"[go_to_chat] {_gtc_exc}\n{_gtc_tb.format_exc()}\n---\n")
        return show_screen4(False, False, False, False, True)

    _btn_outs = [back_user_btn, cancel_train_btn]

    user_qs_start_btn.click(
        go_to_progress, outputs=_all_screens
    ).then(show_cancel_btn, outputs=_btn_outs
    ).then(
        run_user_quickstart,
        inputs=[user_qs_dataset, user_qs_file, user_qs_name, user_qs_test_mode, user_qs_test_steps, user_qs_max_steps],
        outputs=[user_progress_html, user_progress_out],
        show_progress=False,
    ).then(show_home_btn, outputs=_btn_outs
    ).then(go_to_chat, outputs=_all_screens)

    user_cm_start_btn.click(
        go_to_progress, outputs=_all_screens
    ).then(show_cancel_btn, outputs=_btn_outs
    ).then(
        run_user_custommix,
        inputs=[user_cm_model, user_cm_dataset, user_cm_file, user_cm_name, user_cm_test_steps, user_cm_epochs, user_cm_max_steps],
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
    _user_error_log_lines = []

    def _append_error_log(text: str) -> str:
        import datetime as _dt_err
        ts = _dt_err.datetime.now().strftime("%H:%M:%S")
        _user_error_log_lines.insert(0, f"[{ts}] {text}")
        if len(_user_error_log_lines) > 50:
            _user_error_log_lines.pop()
        entry = "\n\n".join(_user_error_log_lines[:10])
        # Also write to disk for persistent inspection
        try:
            with open("logs/ft_chat_errors.log", "a") as _ef:
                _ef.write(f"[{ts}] {text}\n---\n")
        except Exception:
            pass
        return entry

    def _build_chat_content(message: str, image):
        """Build chat history content — plain string for text-only, base64 img+text for image."""
        if image is None:
            return message
        import base64, io
        try:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, "JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            img_html = (f'<img src="data:image/jpeg;base64,{b64}" '
                        f'style="max-width:280px;max-height:200px;border-radius:8px;'
                        f'display:block;margin-bottom:6px;">')
            return f'{img_html}{message}' if message.strip() else img_html
        except Exception:
            return message

    def user_chat_send(message, history, image=None):
        if not message.strip():
            yield history or [], "", "No errors yet." if not _user_error_log_lines else "\n\n".join(_user_error_log_lines[:10]), None
            return
        history = list(history or [])
        with _state._chat_lock:
            _model_ready = _state._chat_model is not None
        if not _model_ready:
            def _is_valid_model(p):
                mp = Path(p) if p else None
                return mp and mp.exists() and (mp / "tokenizer.json").exists() and (mp / "config.json").exists()

            # Priority: last-trained VL export → any VL export → base VL model → any valid model
            # NVFP4/vLLM models are inference-only and cannot be loaded for chat — always skip them
            model_path = _user_last_model_path.get("path")
            if not _is_valid_model(model_path) or not _is_vl_model(model_path or "") or _is_vllm_model(model_path or ""):
                # Search exports for a VL model (newest first)
                exp_dir = ROOT / "exports"
                vl_export = None
                if exp_dir.exists():
                    for d in sorted(exp_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                        candidate = d / "merged"
                        if _is_valid_model(str(candidate)) and _is_vl_model(str(candidate)) and not _is_vllm_model(str(candidate)):
                            vl_export = str(candidate)
                            break
                if vl_export:
                    model_path = vl_export
                else:
                    # Fall back to base VL model from models/ — prefer DEFAULT_BASE_DIR explicitly
                    default_vl = ROOT / "models" / DEFAULT_BASE_DIR
                    if _is_valid_model(str(default_vl)):
                        model_path = str(default_vl)
                    else:
                        models_dir = ROOT / "models"
                        if models_dir.exists():
                            for d in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                                if _is_valid_model(str(d)) and _is_vl_model(str(d)) and not _is_vllm_model(str(d)):
                                    model_path = str(d)
                                    break
                    # Last resort: any valid non-NVFP4 model
                    if not _is_valid_model(model_path):
                        models_dir = ROOT / "models"
                        if models_dir.exists():
                            for d in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                                if _is_valid_model(str(d)) and not _is_vllm_model(str(d)):
                                    model_path = str(d)
                                    break
            if not model_path:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": "No model found. Please complete the training step or add a model to the models/ directory."})
                yield history, "", "No errors yet.", None
                return
            try:
                with open("logs/vl_debug.log", "a") as _vd_mp:
                    _vd_mp.write(f"[model_load] path={model_path} is_vl={_is_vl_model(model_path)}\n")
            except Exception:
                pass
            # Show loading state WITHOUT mutating `history` — loading messages must NOT
            # end up in the conversation context fed to the model (they confuse it).
            _user_content = _build_chat_content(message, image)
            _loading_hist = list(history) + [
                {"role": "user", "content": _user_content},
                {"role": "assistant", "content": "⏳ Loading your model, please wait (~30–60 s)..."},
            ]
            yield _loading_hist, "", "No errors yet.", None
            try:
                _new_tok, _new_model, _new_is_vision = _load_chat_model(model_path)
                with _state._chat_lock:
                    _state._chat_tok, _state._chat_model, _state._chat_is_vision = _new_tok, _new_model, _new_is_vision
                    _state._chat_name = model_path
                _loading_hist[-1]["content"] = "✓ Model loaded. Now answering your message..."
                yield _loading_hist, "", "No errors yet.", None
            except Exception as e:
                import traceback as _tb_load
                _tb_str = _tb_load.format_exc()
                err_log = _append_error_log(f"Model load failed: {e}\n{_tb_str}")
                _loading_hist[-1]["content"] = f"Failed to load model: {e}\n\nSee Error Log for details."
                yield _loading_hist, "", err_log, None
                return
        err_log_val = "No errors yet." if not _user_error_log_lines else "\n\n".join(_user_error_log_lines[:10])
        try:
            if image is not None and not _state._chat_is_vision:
                reply = ("⚠️ This model is text-only and cannot process images. "
                         "Load a vision-language model (e.g. Nemotron 12B VL) to use image input.")
            else:
                import re as _re_hist
                msgs = [{"role": "system", "content": "You are a helpful assistant."}]
                # Only keep last 6 history messages to avoid unbounded context growth.
                # Strip <img> base64 HTML from history — passing raw base64 as text
                # tokenises into ~60K tokens and causes CUDA OOM on the causal mask.
                for t in list(history)[-6:]:
                    if isinstance(t, dict):
                        content = t.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(i.get("text", str(i)) for i in content)
                        content = _re_hist.sub(r'<img\b[^>]*>', '[image]', str(content)).strip()
                        if content:
                            msgs.append({"role": t["role"], "content": content})
                msgs.append({"role": "user", "content": message})
                # 512 for VL (image present) keeps causal mask under ~2 GiB; 256 for text
                _max_tok = 512 if (image is not None and _state._chat_is_vision) else 256
                reply = _generate_reply(msgs, max_tokens=_max_tok, image=image)
        except Exception as e:
            import traceback as _tb_gen
            _tb_str = _tb_gen.format_exc()
            err_reason = type(e).__name__
            if "NoneType" in str(e) and "shape" in str(e):
                err_detail = "Generation failed: model internal error (NoneType.shape). Check Error Log."
            elif "CUDA out of memory" in str(e):
                err_detail = "Generation failed: GPU out of memory. Try restarting the app."
            elif "device" in str(e).lower():
                err_detail = f"Generation failed: device/GPU error — {e}"
            else:
                err_detail = f"Generation failed ({err_reason}): {e}"
            err_log_val = _append_error_log(f"{err_detail}\n\nFull traceback:\n{_tb_str}")
            reply = f"⚠️ {err_detail}\n\nOpen the **Error Log** panel below to see the full traceback."
        history.append({"role": "user", "content": _build_chat_content(message, image)})
        history.append({"role": "assistant", "content": reply})
        yield history, "", err_log_val, None

    user_send_btn.click(user_chat_send,
                        inputs=[user_chat_input, user_chatbot, user_chat_image],
                        outputs=[user_chatbot, user_chat_input, user_error_log, user_chat_image], show_progress=False)
    user_chat_input.submit(user_chat_send,
                           inputs=[user_chat_input, user_chatbot, user_chat_image],
                           outputs=[user_chatbot, user_chat_input, user_error_log, user_chat_image], show_progress=False)
    user_clear_btn.click(lambda: ([], None), outputs=[user_chatbot, user_chat_image])

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
        # H-6: Scope HF login to this publish call only; don't leak token into process env
        import huggingface_hub as _hfhub
        _hfhub.login(token=token.strip(), add_to_git_credential=False)
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
                            ds_file = gr.File(label="Select a file  (.jsonl / .json / .csv / .parquet / .pdf)",
                                              file_types=[".jsonl", ".json", ".csv", ".parquet", ".pdf"],
                                              file_count="single")
                            ds_name_file = gr.Textbox(label="Local name  (optional — defaults to filename)",
                                                       placeholder="leave blank to use filename")
                            ds_pdf_prompt = gr.Textbox(
                                label="PDF page prompt  (only used for PDF uploads)",
                                value="Describe this page in detail.",
                                visible=True,
                            )
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
    
                def do_upload(file, name, pdf_prompt):
                    if file is None:
                        yield terminal_wrap('<span style="color:#f85149;">No file selected.</span>')
                        return
                    fpath = Path(file.name)
                    derived = name.strip() or fpath.stem.lower().replace(" ", "-")
                    if fpath.suffix.lower() == ".pdf":
                        try:
                            import sys as _sys
                            _sys.path.insert(0, str(ROOT))
                            from pdf_utils import pdf_dataset_to_jsonl
                            ds_dir = ROOT / "datasets" / derived
                            ds_dir.mkdir(parents=True, exist_ok=True)
                            yield terminal_wrap(f"Converting PDF to image dataset: <b>{derived}</b>...")
                            jsonl_path = pdf_dataset_to_jsonl(
                                str(fpath), str(ds_dir),
                                prompt=pdf_prompt.strip() or "Describe this page in detail.",
                            )
                            # M-5: write to canonical datasets/<name>/train.jsonl
                            import shutil as _sh
                            train_dir = ROOT / "datasets" / derived
                            train_dir.mkdir(parents=True, exist_ok=True)
                            _sh.copy2(jsonl_path, str(train_dir / "train.jsonl"))
                            n_lines = sum(1 for _ in open(jsonl_path))
                            yield terminal_wrap(
                                f'<span style="color:#3fb950;">PDF converted: {n_lines} pages → '
                                f'<b>{derived}</b> (ready for multimodal training)</span>')
                        except Exception as exc:
                            yield terminal_wrap(
                                f'<span style="color:#f85149;">PDF conversion failed: {exc}</span>')
                        return
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
                add_ds_file_btn.click(do_upload, inputs=[ds_file, ds_name_file, ds_pdf_prompt],
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

                        # Training mode — auto-detected from model; user can override
                        cfg_mode = gr.Radio(
                            ["text", "multimodal"],
                            value="text",
                            label="Training mode",
                            info="Auto-set when model is selected. Multimodal uses VL trainer.",
                        )

                        # Quantization — NVFP4 is INFERENCE-ONLY, never offered here
                        cfg_quant = gr.Radio(
                            ["BF16 + LoRA", "8-bit + LoRA", "4-bit QLoRA NF4"],
                            value="BF16 + LoRA",
                            label="Training quantization",
                            info="NVFP4 is inference-only and is not available for training.",
                        )

                        # Multimodal-only options (shown when mode=multimodal)
                        with gr.Group(visible=False) as cfg_vl_group:
                            cfg_lora_r = gr.Slider(8, 64, value=16, step=8, label="LoRA rank")
                            cfg_img_res = gr.Radio(
                                ["224", "448", "672"],
                                value="224",
                                label="Image resolution (px)",
                            )

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

                        # Auto-detect training mode when model is chosen
                        def _auto_detect_mode(model_name):
                            if not model_name:
                                return gr.Radio(value="text"), gr.Group(visible=False)
                            p = ROOT / "models" / model_name
                            is_vl = _is_vl_model(str(p))
                            return gr.Radio(value="multimodal" if is_vl else "text"), gr.Group(visible=is_vl)
                        cfg_model.change(_auto_detect_mode, inputs=cfg_model,
                                         outputs=[cfg_mode, cfg_vl_group])
                        cfg_mode.change(
                            lambda m: gr.Group(visible=(m == "multimodal")),
                            inputs=cfg_mode, outputs=cfg_vl_group)

                        with gr.Row():
                            gr.Button("↺ Refresh", size="sm", variant="secondary").click(
                                lambda: (gr.Dropdown(choices=list_models()),
                                         gr.Dropdown(choices=list_cleaned_datasets())),
                                outputs=[cfg_model, cfg_ds])
                            cfg_btn = gr.Button("Generate / Save config", variant="primary")

                    with gr.Column(scale=2):
                        cfg_out  = gr.HTML(terminal_wrap(""))
                        cfg_yaml = gr.Code(label="Generated YAML (text mode) / Config summary (VL mode)",
                                           language="yaml", lines=22, interactive=False)

                def do_configure(model, ds, mode, quant, lora_r, lr, epochs, seq,
                                 wandb_on, wandb_project, wandb_entity):
                    if mode == "multimodal":
                        # For VL mode, write a JSON config file instead of Axolotl YAML
                        import datetime
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        cfg_name = f"vl_{ts}.json"
                        quant_map = {
                            "BF16 + LoRA": "bf16",
                            "8-bit + LoRA": "8bit",
                            "4-bit QLoRA NF4": "4bit",
                        }
                        cfg_data = {
                            "mode": "multimodal",
                            "model": model,
                            "dataset": ds,
                            "quant": quant_map.get(quant, "bf16"),
                            "lora_r": int(lora_r),
                            "lr": lr,
                            "epochs": int(epochs),
                            "seq_len": int(seq),
                        }
                        cfg_path = ROOT / "configs" / cfg_name
                        cfg_path.parent.mkdir(exist_ok=True)
                        cfg_path.write_text(json.dumps(cfg_data, indent=2))
                        summary = json.dumps(cfg_data, indent=2)
                        yield terminal_wrap(
                            f'<span style="color:#3fb950;">VL config saved: {cfg_name}</span>'), summary
                        return

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
                              inputs=[cfg_model, cfg_ds, cfg_mode, cfg_quant, cfg_lora_r,
                                      cfg_lr, cfg_epochs, cfg_seq,
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

                def _latest_vl_config():
                    """Return parsed VL JSON config dict if the latest config is multimodal, else None."""
                    cfgs = list_configs()
                    if not cfgs:
                        return None
                    c = cfgs[0]
                    cfg_path = ROOT / "configs" / c
                    if not cfg_path.exists():
                        return None
                    try:
                        data = json.loads(cfg_path.read_text())
                        if data.get("mode") == "multimodal":
                            return data
                    except Exception:
                        pass
                    return None

                def _vl_train_cmd(cfg_data, smoke=False):
                    """Build the vl_trainer.py subprocess command from a VL config dict."""
                    quant_map = {
                        "BF16 + LoRA": "bf16",
                        "8-bit + LoRA": "8bit",
                        "4-bit QLoRA NF4": "4bit",
                        "bf16": "bf16", "8bit": "8bit", "4bit": "4bit",
                    }
                    model_name = cfg_data.get("model", "")
                    model_path = str(ROOT / "models" / model_name) if "/" not in model_name else model_name
                    ds_name    = cfg_data.get("dataset", "")
                    # M-5: canonical path is datasets/<name>; cleaned_datasets is legacy
                    if "/" not in ds_name:
                        _ds_canonical = ROOT / "datasets" / ds_name
                        _ds_legacy    = ROOT / "cleaned_datasets" / ds_name
                        ds_path = str(_ds_canonical if _ds_canonical.exists() else _ds_legacy)
                    else:
                        ds_path = ds_name
                    out_path   = str(ROOT / "checkpoints" / f"vl_{model_name}")
                    quant      = quant_map.get(cfg_data.get("quant", "bf16"), "bf16")
                    lora_r     = cfg_data.get("lora_r", 16)
                    lr         = cfg_data.get("lr", 2e-4)
                    epochs     = cfg_data.get("epochs", 3)
                    cmd = [
                        "python", str(ROOT / "vl_trainer.py"),
                        "--model", model_path,
                        "--dataset", ds_path,
                        "--output", out_path,
                        "--quant", quant,
                        "--lora-r", str(lora_r),
                        "--lr", str(lr),
                        "--epochs", str(int(epochs)),
                    ]
                    if smoke:
                        cmd.append("--smoke")
                    return cmd

                def do_smoke(steps):
                    c = _latest_config()
                    if not c:
                        yield gr.Button(visible=False), terminal_wrap(
                            '<span style="color:#f85149;">No recipe found — run Configure first.</span>')
                        return
                    vl_cfg = _latest_vl_config()
                    if vl_cfg:
                        import subprocess as _sp
                        cmd = _vl_train_cmd(vl_cfg, smoke=True)
                        yield gr.Button(visible=True), terminal_wrap(f"Running VL smoke test ({c})...")
                        with _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
                                       env={**os.environ,
                                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                            "HF_HUB_OFFLINE": "1",
                                            "TOKENIZERS_PARALLELISM": "false"}) as proc:
                            buf = []
                            for line in proc.stdout:
                                buf.append(line.rstrip())
                                yield gr.Button(visible=True), terminal_wrap("\n".join(buf[-200:]))
                        chunk = terminal_wrap("\n".join(buf[-200:]))
                        yield gr.Button(visible=False), chunk
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
                    vl_cfg = _latest_vl_config()
                    if vl_cfg:
                        import subprocess as _sp
                        cmd = _vl_train_cmd(vl_cfg, smoke=False)
                        if int(max_steps or 0) > 0:
                            cmd += ["--max-steps", str(int(max_steps))]
                        yield gr.Button(visible=True), terminal_wrap(f"Starting VL training ({c})...")
                        with _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True,
                                       env={**os.environ,
                                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                                            "HF_HUB_OFFLINE": "1",
                                            "TOKENIZERS_PARALLELISM": "false"}) as proc:
                            buf = []
                            for line in proc.stdout:
                                buf.append(line.rstrip())
                                yield gr.Button(visible=True), terminal_wrap("\n".join(buf[-200:]))
                        chunk = terminal_wrap("\n".join(buf[-200:]))
                        yield gr.Button(visible=False), chunk
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
                    _sp.run(["pkill", "-f", "vl_trainer.py"],     capture_output=True)
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
                            # H-6: scope login to this call; don't pollute the process env
                            import huggingface_hub as _hfhub
                            _hfhub.login(token=token.strip(), add_to_git_credential=False)
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
                        chatbot = gr.Chatbot(height=380)
                        chat_image = gr.Image(
                            type="pil",
                            label="Upload image (optional — for vision models)",
                            height=140)
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Type your message...",
                                label="", scale=4, lines=1,
                            )
                            chat_send  = gr.Button("Send", variant="primary", scale=0)
                        chat_clear = gr.Button("Clear conversation", size="sm", variant="secondary")
    
                def load_model(model_name):
                    if not model_name:
                        return '<div style="color:#f85149;padding:8px;">Select a model first.</div>'
                    with _state._chat_lock:
                        _old_model = _state._chat_model
                        _state._chat_model = None
                    if _old_model is not None:
                        del _old_model
                        import gc; gc.collect()
                        try:
                            import torch; torch.cuda.empty_cache()
                        except Exception:
                            pass
                    _stop_vllm_server()

                    resolved = resolve_chat_path(model_name)
                    model_path = Path(resolved)
                    if not model_path.exists():
                        model_path = ROOT / resolved
                    if not model_path.exists():
                        return f'<div style="color:#f85149;padding:8px;">Path not found: {resolved}</div>'

                    try:
                        _new_tok, _new_model, _new_is_vision = _load_chat_model(str(model_path))
                        with _state._chat_lock:
                            _state._chat_tok, _state._chat_model, _state._chat_is_vision = _new_tok, _new_model, _new_is_vision
                            _state._chat_name = model_name
                        with _state._vllm_lock:
                            _client = _state._vllm_client
                        with _state._chat_lock:
                            _is_vis = _state._chat_is_vision
                        label = model_name.split("  [")[0]
                        if _client is not None:
                            tag = " (vLLM · omni)"
                        elif _is_vis:
                            tag = " (vision)"
                        else:
                            tag = ""
                        return (f'<div style="color:#3fb950;font-weight:600;padding:8px;">'
                                f'Loaded: {label}{tag}</div>')
                    except Exception as e:
                        return f'<div style="color:#f85149;padding:8px;">Failed: {e}</div>'

                def unload_model():
                    _stop_vllm_server()
                    with _state._chat_lock:
                        _old_model = _state._chat_model
                        _old_tok   = _state._chat_tok
                        _state._chat_model = None
                        _state._chat_tok   = None
                        _state._chat_name  = None
                    if _old_model is not None:
                        del _old_model
                    if _old_tok is not None:
                        del _old_tok
                    import gc; gc.collect()
                    try:
                        import torch; torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return '<div style="color:#8b949e;padding:8px;">Model unloaded. GPU memory freed.</div>'
    
                def chat_respond(message, history, system_prompt, max_tokens, image=None):
                    if not message.strip():
                        yield history or [], ""
                        return
                    history = list(history or [])
                    if _state._chat_model is None and _state._vllm_client is None:
                        history.append({"role": "user",      "content": message})
                        history.append({"role": "assistant", "content": "No model loaded. Please load a model first."})
                        yield history, ""
                        return
                    try:
                        def extract_text(content) -> str:
                            if isinstance(content, str):
                                return content
                            if isinstance(content, list):
                                return " ".join(i.get("text", str(i)) if isinstance(i, dict) else str(i) for i in content)
                            return str(content)

                        if image is not None and not _state._chat_is_vision:
                            reply = ("⚠️ This model is text-only and cannot process images. "
                                     "Load a vision-language model (e.g. Nemotron 12B VL) to use image input.")
                        else:
                            msgs = [{"role": "system", "content": system_prompt}]
                            for turn in history:
                                if isinstance(turn, dict):
                                    msgs.append({"role": turn.get("role", "user"),
                                                 "content": extract_text(turn.get("content", ""))})
                            msgs.append({"role": "user", "content": message})
                            reply = _generate_reply(msgs, max_tokens=max_tokens, image=image)
                    except Exception as e:
                        reply = f"Error: {e}"

                    history.append({"role": "user",      "content": message})
                    history.append({"role": "assistant", "content": reply})
                    yield history, ""

                chat_load_btn.click(load_model, inputs=chat_model_dd, outputs=chat_status)
                chat_unload_btn.click(unload_model, outputs=chat_status)
                chat_send.click(chat_respond,
                                inputs=[chat_input, chatbot, chat_system, chat_max_tokens, chat_image],
                                outputs=[chatbot, chat_input], show_progress=False)
                chat_input.submit(chat_respond,
                                  inputs=[chat_input, chatbot, chat_system, chat_max_tokens, chat_image],
                                  outputs=[chatbot, chat_input], show_progress=False)
                chat_clear.click(lambda: ([], None), outputs=[chatbot, chat_image])
    
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
    ap = argparse.ArgumentParser(
        description="MKD Fine-Tuning Platform. "
                    "Set ADMIN_PASSWORD env var to enable auth. "
                    "To expose externally use --host 0.0.0.0 (or a reverse proxy)."
    )
    ap.add_argument("--port",  type=int, default=7860)
    # H-7: Default to localhost-only for security.
    # Use --host 0.0.0.0 to expose on the network (behind a firewall / reverse proxy).
    ap.add_argument("--host",  default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    # H-7: Optional HTTP basic auth via ADMIN_PASSWORD env var
    _admin_pw = os.environ.get("ADMIN_PASSWORD", "").strip()
    _auth     = [("admin", _admin_pw)] if _admin_pw else None

    print(f"\n   MKD Fine-Tuning Platform")
    print(f" Project : {ROOT}")
    print(f" URL     : http://{args.host}:{args.port}")
    if _auth:
        print(" Auth    : basic auth enabled (username: admin)")
    else:
        print(" Auth    : WARNING — ADMIN_PASSWORD not set, UI is unprotected.")
    print()

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
        auth=_auth,
        theme=theme,
        css=CSS,
        max_threads=10,
    )
