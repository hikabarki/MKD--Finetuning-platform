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

# vLLM may live in a separate venv. Set $VLLM_PYTHON to point at it; otherwise
# default to the interpreter running the platform (works when vllm is installed
# in the platform's own venv — the recommended setup on a fresh server).
VLLM_PYTHON = os.environ.get("VLLM_PYTHON", sys.executable)
VLLM_PORT   = int(os.environ.get("VLLM_PORT", "8001"))

# ── Project root ──────────────────────────────────────────────────────────────
# __file__ is inference/state.py, so .parent.parent = project root
ROOT    = Path(os.environ.get("FT_PROJECT", Path(__file__).resolve().parent.parent))
# SCRIPTS always points to the platform code directory (not the project data dir)
_PLATFORM = Path(__file__).resolve().parent.parent
SCRIPTS = _PLATFORM / "scripts"
PYTHON  = sys.executable

# ── Default model for General User Mode ───────────────────────────────────────
DEFAULT_BASE_MODEL = "google/gemma-4-12b"
DEFAULT_BASE_DIR   = "gemma-4-12b"
_FALLBACK_BASE_DIR = "gemma-4-12b"
MODEL_HF_ID = "google/gemma-4-12b"

# Pin to a specific HF Hub commit to prevent silent supply-chain updates.
# Set MODEL_REVISION_SHA env var to a full 40-char commit SHA; None = "main".
REVISION_SHA: str | None = os.environ.get("MODEL_REVISION_SHA")

_MODEL_DISPLAY_INFO: dict = {
    "gemma-4-12b": "Gemma 4 12B · image+text  (23 GB VRAM)",
}
