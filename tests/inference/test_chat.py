"""Tests for inference/chat.py — pure-Python functions (no GPU required)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── _is_vision_model ──────────────────────────────────────────────────────────

def test_is_vision_model_preprocessor_config(tmp_path: Path):
    (tmp_path / "preprocessor_config.json").write_text("{}")
    from inference.chat import _is_vision_model
    assert _is_vision_model(str(tmp_path)) is True


def test_is_vision_model_processor_config(tmp_path: Path):
    (tmp_path / "processor_config.json").write_text("{}")
    from inference.chat import _is_vision_model
    assert _is_vision_model(str(tmp_path)) is True


def test_is_vision_model_text_only(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    from inference.chat import _is_vision_model
    assert _is_vision_model(str(tmp_path)) is False


def test_is_vision_model_empty_dir(tmp_path: Path):
    from inference.chat import _is_vision_model
    assert _is_vision_model(str(tmp_path)) is False


# ── _is_vl_model ──────────────────────────────────────────────────────────────

def test_is_vl_model_auto_map_vl(tmp_path: Path):
    cfg = {"auto_map": {"AutoModel": "modeling_vl.NemotronVLModel"}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    from inference.chat import _is_vl_model
    assert _is_vl_model(str(tmp_path)) is True


def test_is_vl_model_vision_config(tmp_path: Path):
    cfg = {"vision_config": {"model_type": "radio"}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    from inference.chat import _is_vl_model
    assert _is_vl_model(str(tmp_path)) is True


def test_is_vl_model_vl_architecture(tmp_path: Path):
    cfg = {"architectures": ["InternVLForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    from inference.chat import _is_vl_model
    assert _is_vl_model(str(tmp_path)) is True


def test_is_vl_model_text_architecture(tmp_path: Path):
    cfg = {"architectures": ["LlamaForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    from inference.chat import _is_vl_model
    assert _is_vl_model(str(tmp_path)) is False


def test_is_vl_model_no_config(tmp_path: Path):
    from inference.chat import _is_vl_model
    assert _is_vl_model(str(tmp_path)) is False


def test_is_vl_model_corrupt_config(tmp_path: Path):
    (tmp_path / "config.json").write_text("NOT JSON{{{")
    from inference.chat import _is_vl_model
    assert _is_vl_model(str(tmp_path)) is False


# ── _nvfp4_quant_config ───────────────────────────────────────────────────────

def test_nvfp4_quant_config_no_cuda(monkeypatch):
    """Without CUDA, returns None gracefully."""
    from inference.chat import _nvfp4_quant_config
    with patch.dict("sys.modules", {"torch": None}):
        result = _nvfp4_quant_config()
    assert result is None


def test_nvfp4_quant_config_old_gpu():
    """SM < 9.x → returns None (not Blackwell)."""
    from inference.chat import _nvfp4_quant_config
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (8, 0)
    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": MagicMock()}):
        result = _nvfp4_quant_config()
    assert result is None


def test_nvfp4_quant_config_blackwell():
    """SM ≥ 9.x → returns a BitsAndBytesConfig."""
    from inference.chat import _nvfp4_quant_config
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_capability.return_value = (9, 0)
    mock_torch.bfloat16 = "bfloat16"

    mock_bnb = MagicMock()
    mock_transformers = MagicMock()
    mock_transformers.BitsAndBytesConfig = mock_bnb

    with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
        result = _nvfp4_quant_config()

    mock_bnb.assert_called_once()
    call_kwargs = mock_bnb.call_args.kwargs
    assert call_kwargs.get("load_in_4bit") is True
    assert call_kwargs.get("bnb_4bit_quant_type") == "fp4"
