"""
inference/vllm_server.py — vLLM server lifecycle management.
"""
from __future__ import annotations

import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from inference import state as _state


def _is_vllm_model(path: str) -> bool:
    """True if the model needs vLLM (pre-quantized NVFP4/FP8 not loadable by transformers)."""
    p = Path(path)
    if (p / "hf_quant_config.json").exists():
        return True
    cfg = p / "config.json"
    if cfg.exists():
        import json
        try:
            c = json.loads(cfg.read_text())
            arch = c.get("architectures", [""])[0]
            vllm_archs = {
                "NemotronH_Nano_Omni_Reasoning_V3",
                "NemotronH_Super_Omni_Reasoning_V3",
            }
            if arch in vllm_archs:
                return True
        except Exception:
            pass
    return False


def _start_vllm_server(model_path: str) -> bool:
    """Launch vLLM OpenAI-compatible server for the given model. Returns True on success."""
    _stop_vllm_server()

    cmd = [
        _state.VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--trust-remote-code",
        "--port", str(_state.VLLM_PORT),
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.85",
        "--enforce-eager",
    ]
    _proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with _state._vllm_lock:
        _state._vllm_proc    = _proc
        _state._vllm_model_id = model_path

    health_url = f"http://localhost:{_state.VLLM_PORT}/health"
    for _ in range(180):
        time.sleep(1)
        try:
            urllib.request.urlopen(health_url, timeout=2)
            break
        except Exception:
            if _proc.poll() is not None:
                return False
    else:
        return False

    try:
        from openai import OpenAI
        _client = OpenAI(base_url=f"http://localhost:{_state.VLLM_PORT}/v1", api_key="none")
    except ImportError:
        subprocess.run([_state.VLLM_PYTHON, "-m", "pip", "install", "openai", "-q"])
        from openai import OpenAI
        _client = OpenAI(base_url=f"http://localhost:{_state.VLLM_PORT}/v1", api_key="none")
    with _state._vllm_lock:
        _state._vllm_client = _client
    return True


def _stop_vllm_server():
    """Terminate the running vLLM server process AND any orphaned vLLM processes."""
    with _state._vllm_lock:
        _proc = _state._vllm_proc
        _state._vllm_proc = None
    if _proc is not None:
        try:
            _proc.terminate()
            _proc.wait(timeout=10)
        except Exception:
            try:
                _proc.kill()
            except Exception:
                pass
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "EngineCore"],        capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "VllmWorker"],        capture_output=True)
    time.sleep(3)  # Give OS time to reclaim VRAM
    with _state._vllm_lock:
        _state._vllm_client   = None
        _state._vllm_model_id = None
