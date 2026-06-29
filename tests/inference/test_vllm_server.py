"""Tests for inference/vllm_server.py — vLLM lifecycle logic."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_is_vllm_model_hf_quant_config(tmp_path: Path):
    """Model with hf_quant_config.json → vLLM required."""
    (tmp_path / "hf_quant_config.json").write_text("{}")
    from inference.vllm_server import _is_vllm_model
    assert _is_vllm_model(str(tmp_path)) is True


def test_is_vllm_model_quant_method(tmp_path: Path):
    """Model with fp8/nvfp4 quantization_config → vLLM required."""
    cfg = {"architectures": ["Gemma4ForConditionalGeneration"],
           "quantization_config": {"quant_method": "fp8"}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    from inference.vllm_server import _is_vllm_model
    assert _is_vllm_model(str(tmp_path)) is True


def test_is_vllm_model_arch_only_not_vllm(tmp_path: Path):
    """Architecture alone (no quant config) → NOT vLLM (fine-tuned exports use Transformers)."""
    cfg = {"architectures": ["Gemma4ForConditionalGeneration"]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    from inference.vllm_server import _is_vllm_model
    assert _is_vllm_model(str(tmp_path)) is False


def test_is_vllm_model_normal_model(tmp_path: Path):
    """Standard model → not a vLLM model."""
    cfg = {"architectures": ["LlamaForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    from inference.vllm_server import _is_vllm_model
    assert _is_vllm_model(str(tmp_path)) is False


def test_is_vllm_model_empty_dir(tmp_path: Path):
    """Empty directory → not a vLLM model."""
    from inference.vllm_server import _is_vllm_model
    assert _is_vllm_model(str(tmp_path)) is False


def test_is_vllm_model_bad_config(tmp_path: Path):
    """Corrupt config.json → falls back to False gracefully."""
    (tmp_path / "config.json").write_text("NOT JSON{{{")
    from inference.vllm_server import _is_vllm_model
    assert _is_vllm_model(str(tmp_path)) is False


def test_stop_vllm_server_cleans_state():
    """_stop_vllm_server should clear proc/client/model_id from state."""
    import inference.state as _state
    from inference.vllm_server import _stop_vllm_server

    mock_proc = MagicMock()
    mock_proc.terminate = MagicMock()
    mock_proc.wait      = MagicMock()

    with _state._vllm_lock:
        _state._vllm_proc    = mock_proc
        _state._vllm_client  = MagicMock()
        _state._vllm_model_id = "/some/model"

    with patch("subprocess.run"):
        with patch("time.sleep"):
            _stop_vllm_server()

    assert _state._vllm_proc    is None
    assert _state._vllm_client  is None
    assert _state._vllm_model_id is None
    mock_proc.terminate.assert_called_once()


def test_stop_vllm_server_no_proc():
    """_stop_vllm_server with no running process should not raise."""
    import inference.state as _state
    from inference.vllm_server import _stop_vllm_server

    with _state._vllm_lock:
        _state._vllm_proc = None

    with patch("subprocess.run"):
        with patch("time.sleep"):
            _stop_vllm_server()   # must not raise


def test_start_vllm_server_process_dies_immediately():
    """If the process exits immediately, _start_vllm_server returns False."""
    import inference.state as _state
    from inference.vllm_server import _start_vllm_server

    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1   # process already dead

    with patch("subprocess.Popen", return_value=mock_proc), \
         patch("subprocess.run"), \
         patch("time.sleep"), \
         patch("urllib.request.urlopen", side_effect=OSError("refused")):
        result = _start_vllm_server("/fake/model")

    assert result is False
