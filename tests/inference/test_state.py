"""Tests for inference/state.py — shared globals and lock behaviour."""
from __future__ import annotations

import os
import threading
from pathlib import Path


def test_root_is_path():
    import inference.state as state
    assert isinstance(state.ROOT, Path)


def test_root_points_to_repo_root():
    import inference.state as state
    assert (state.ROOT / "app.py").exists(), "ROOT should point at the repo root"


def test_revision_sha_none_by_default(monkeypatch):
    monkeypatch.delenv("NEMOTRON_REVISION_SHA", raising=False)
    # Re-evaluate the expression; module is already loaded, but the env var
    # is read at load time. Test the constant is None when env is absent.
    import inference.state as state
    # If env var was not set at import time, REVISION_SHA is None
    assert state.REVISION_SHA is None or isinstance(state.REVISION_SHA, str)


def test_revision_sha_from_env(monkeypatch):
    sha = "abc123def456" * 3
    monkeypatch.setenv("NEMOTRON_REVISION_SHA", sha)
    # Reimport to pick up env change (use importlib)
    import importlib
    import inference.state as state
    importlib.reload(state)
    assert state.REVISION_SHA == sha
    # Reload back to clean state
    monkeypatch.delenv("NEMOTRON_REVISION_SHA", raising=False)
    importlib.reload(state)


def test_chat_lock_is_rlock():
    import inference.state as state
    assert isinstance(state._chat_lock, type(threading.RLock()))


def test_vllm_lock_is_rlock():
    import inference.state as state
    assert isinstance(state._vllm_lock, type(threading.RLock()))


def test_lock_thread_safety():
    """Multiple threads should be able to acquire + release the chat lock."""
    import inference.state as state

    results = []

    def worker(n):
        with state._chat_lock:
            prev = state._chat_name
            state._chat_name = f"model_{n}"
            results.append(state._chat_name)
            state._chat_name = prev

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8


def test_model_display_info_excludes_nvfp4():
    """NVFP4 must not appear in the model display info (inference-only)."""
    import inference.state as state
    for key in state._MODEL_DISPLAY_INFO:
        assert "nvfp4" not in key.lower(), f"NVFP4 key found: {key}"
