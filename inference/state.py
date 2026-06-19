"""
inference/state.py — Shared mutable state for all inference threads.

All module-level variables here are intentionally mutable globals.
Other modules should import this module (not star-import it) and access
attributes as state._chat_model, state._vllm_proc, etc.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# ── Chat model state (loaded lazily) ─────────────────────────────────────────
_chat_model     = None
_chat_tok       = None   # AutoTokenizer (text) or AutoProcessor (vision)
_chat_name      = None
_chat_is_vision = False  # True when a VLM (image-capable) model is loaded

# vLLM server backend (used for pre-quantized NVFP4 models)
_vllm_proc      = None   # subprocess.Popen
_vllm_client    = None   # openai.OpenAI pointing at local vLLM server
_vllm_model_id  = None   # model path string passed to vLLM

# Threading locks for shared mutable state accessed from multiple Gradio threads
_chat_lock             = threading.RLock()   # guards _chat_model/tok/name/is_vision
_vllm_lock             = threading.RLock()   # guards _vllm_proc/client/model_id
_user_model_path_lock  = threading.RLock()
_user_training_lock    = threading.RLock()

VLLM_PYTHON = "/home/mkd-s002/vLLM-h/keural-rag/venv/bin/python"
VLLM_PORT   = 8001

# ── Project root ──────────────────────────────────────────────────────────────
# __file__ is inference/state.py, so .parent.parent = project root
ROOT    = Path(os.environ.get("FT_PROJECT", Path(__file__).resolve().parent.parent))
SCRIPTS = ROOT / "scripts"
PYTHON  = sys.executable

# ── Default model for General User Mode ───────────────────────────────────────
DEFAULT_BASE_MODEL = "nvidia/NVIDIA-Nemotron-Nano-VL-12B-v2"
DEFAULT_BASE_DIR   = "nvidia__nvidia-nemotron-nano-12b-v2-vl-bf16"
_FALLBACK_BASE_DIR = "nvidia__nvidia-nemotron-3-nano-4b-bf16"
NEMOTRON_HF_ID     = "nvidia/NVIDIA-Nemotron-Nano-VL-12B-v2"

# Pin to a specific HF Hub commit to prevent silent supply-chain updates.
# Set NEMOTRON_REVISION_SHA env var to a full 40-char commit SHA; None = "main".
REVISION_SHA: str | None = os.environ.get("NEMOTRON_REVISION_SHA")

_MODEL_DISPLAY_INFO: dict = {
    "nvidia__nvidia-nemotron-nano-12b-v2-vl-bf16": "Nemotron VL 12B · image+text  (25 GB VRAM)",
    "nvidia__nvidia-nemotron-3-nano-4b-bf16":       "Nemotron 4B · BF16  (4.3 GB VRAM)",
}
