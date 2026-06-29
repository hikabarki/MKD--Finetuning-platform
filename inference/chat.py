"""
inference/chat.py — Chat model loading, input building, and text generation.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from inference import state as _state
from inference.vllm_server import _is_vllm_model, _start_vllm_server, _stop_vllm_server


def _is_vision_model(path: str) -> bool:
    """True if the model directory has a vision processor config."""
    p = Path(path)
    return any((p / f).exists() for f in (
        "preprocessor_config.json", "processor_config.json"))


def _is_vl_model(path: str) -> bool:
    """True if the model is a Vision-Language model that needs the VL training path."""
    p = Path(path)
    cfg = p / "config.json"
    if not cfg.exists():
        return False
    try:
        data = json.loads(cfg.read_text())
        auto_map = data.get("auto_map", {})
        for cls in auto_map.values():
            cls_lower = cls.lower()
            if "vl" in cls_lower or "vision" in cls_lower:
                return True
        if "vision_config" in data:
            return True
        arch = data.get("architectures", [])
        return any("vl" in a.lower() or "vision" in a.lower() for a in arch)
    except Exception:
        return False


def _nvfp4_quant_config():
    """BitsAndBytesConfig with FP4 on Blackwell GPUs (SM ≥ 9.x), else None."""
    try:
        import torch
        from transformers import BitsAndBytesConfig
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 9:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
    except Exception:
        pass
    return None


def _load_chat_model(model_path: str):
    """Unified loader for text-only models, VLMs, and pre-quantized vLLM models.
    Returns (tokenizer_or_processor, model, is_vision).
    For vLLM models returns (None, None, True) and sets _state._vllm_client."""
    if _is_vllm_model(model_path):
        ok = _start_vllm_server(model_path)
        if not ok:
            raise RuntimeError(
                "vLLM server failed to start. Check that "
                f"{_state.VLLM_PYTHON} is available and the model path is correct.")
        return None, None, True

    import torch
    from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
    try:
        from transformers import AutoModelForImageTextToText as _VLMClass
    except ImportError:
        try:
            from transformers import AutoModelForVision2Seq as _VLMClass
        except ImportError:
            _VLMClass = AutoModelForCausalLM

    mp = Path(model_path)
    is_vl     = _is_vl_model(model_path)
    is_vision = is_vl or _is_vision_model(model_path)

    if is_vl:
        import glob
        import shutil as _sh
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        _stop_vllm_server()
        subprocess.run(["pkill", "-9", "-f", "vl_trainer.py"],        capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "axolotl.cli.train"],    capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "ft_train.py"],          capture_output=True)
        for _ in range(30):
            torch.cuda.empty_cache()
            if torch.cuda.mem_get_info()[0] / 1024 ** 3 >= 25.0:
                break
            time.sleep(1)

        cache_base = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
        model_name_san = mp.name.replace("-", "_hyphen_")
        cache_model_dir = os.path.join(cache_base, model_name_san)
        if os.path.isdir(cache_model_dir):
            all_py = glob.glob(str(mp / "*.py"))
            for hash_dir in os.listdir(cache_model_dir):
                hash_path = os.path.join(cache_model_dir, hash_dir)
                if os.path.isdir(hash_path) and not hash_dir.startswith("__"):
                    for pyf in all_py:
                        dest = os.path.join(hash_path, os.path.basename(pyf))
                        if not os.path.exists(dest):
                            _sh.copy2(pyf, dest)

        tok = AutoProcessor.from_pretrained(str(mp), trust_remote_code=True, local_files_only=True,
                                             revision=_state.REVISION_SHA)
        try:
            import flash_attn  # noqa: F401
            _attn_impl = "flash_attention_2"
            print(f"Using FlashAttention-2 (v{flash_attn.__version__}).", flush=True)
        except Exception:
            # SDPA with cuDNN backend can fail with CUDNN_STATUS_NOT_INITIALIZED
            # on some environments. Disable cuDNN SDPA and let PyTorch pick
            # flash_sdp or math fallback — still much faster than eager.
            torch.backends.cuda.enable_cudnn_sdp(False)
            _attn_impl = "sdpa"
            print("Using PyTorch SDPA (cuDNN disabled, flash_sdp/math).", flush=True)
        # Keep chat model on a single GPU — the 12B model fits in 24 GB and
        # single-GPU inference is faster than cross-GPU pipeline parallelism.
        vl_load_kw = dict(
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation=_attn_impl,
            revision=_state.REVISION_SHA,
        )
        try:
            import transformers as _tf
            from packaging.version import Version
            if Version(_tf.__version__) >= Version("5.0"):
                vl_load_kw["dtype"] = torch.bfloat16
            else:
                vl_load_kw["torch_dtype"] = torch.bfloat16
        except Exception:
            vl_load_kw["dtype"] = torch.bfloat16

        # Disable transformers 5.x threaded weight loading
        try:
            import transformers.core_model_loading as _cml
            _cml.GLOBAL_WORKERS = 1
        except Exception:
            pass
        import logging as _logging
        _tf_logger = _logging.getLogger("transformers")
        _tf_prev_level = _tf_logger.level
        _tf_logger.setLevel(_logging.ERROR)
        model = AutoModelForCausalLM.from_pretrained(str(mp), **vl_load_kw)
        _tf_logger.setLevel(_tf_prev_level)

        try:
            _rm = model.vision_model.radio_model
            if hasattr(_rm, "summary_idxs") and _rm.summary_idxs is not None:
                _n = _rm.summary_idxs.shape[0]
                _rm.summary_idxs = torch.arange(_n, dtype=torch.int64,
                                                device=_rm.summary_idxs.device)
        except Exception:
            pass

        return tok, model, True

    bnb_cfg = _nvfp4_quant_config()
    load_kw: dict = dict(device_map="auto", trust_remote_code=True, local_files_only=True,
                         revision=_state.REVISION_SHA)
    if bnb_cfg:
        load_kw["quantization_config"] = bnb_cfg
    else:
        try:
            import transformers as _tf
            from packaging.version import Version
            dtype_key = "dtype" if Version(_tf.__version__) >= Version("5.0") else "torch_dtype"
        except Exception:
            dtype_key = "dtype"
        load_kw[dtype_key] = torch.bfloat16

    if is_vision:
        tok = AutoProcessor.from_pretrained(str(mp), trust_remote_code=True, local_files_only=True,
                                             revision=_state.REVISION_SHA)
        try:
            model = _VLMClass.from_pretrained(str(mp), **load_kw)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(str(mp), **load_kw)
    else:
        tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True, local_files_only=True,
                                             revision=_state.REVISION_SHA)
        model = AutoModelForCausalLM.from_pretrained(str(mp), **load_kw)
    return tok, model, is_vision


def _apply_chat_template(tok, messages, tokenize=False, add_generation_prompt=True):
    """Call apply_chat_template on tok, falling back through multiple strategies.

    Strategies (in order):
    1. tok.apply_chat_template()  — standard tokenizer / processor
    2. tok.tokenizer.apply_chat_template()  — Gemma4UnifiedProcessor wraps inner tokenizer
    3. Gemma 4 format  — uses sot_token/eot_token (<|turn> / <turn|>)
    4. Gemma 2/3 format  — <start_of_turn> / <end_of_turn> fallback
    """
    kwargs = dict(tokenize=tokenize, add_generation_prompt=add_generation_prompt)

    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(messages, **kwargs)
        except (ValueError, AttributeError):
            pass

    inner = getattr(tok, "tokenizer", None)
    if inner is not None and hasattr(inner, "apply_chat_template"):
        try:
            return inner.apply_chat_template(messages, **kwargs)
        except (ValueError, AttributeError):
            pass

    # Detect Gemma 4 format via sot_token/eot_token on inner tokenizer.
    # Gemma 4 uses <|turn>role\ncontent<turn|> instead of <start_of_turn>.
    _inner = inner or tok
    sot = getattr(_inner, "sot_token", None)
    eot = getattr(_inner, "eot_token", None)
    bos = getattr(_inner, "bos_token", None) or "<bos>"

    if sot and eot:
        # Gemma 4 format: <bos><|turn>role\ncontent<turn|>...
        parts = [bos]
        for msg in messages:
            role      = msg.get("role", "user")
            content   = str(msg.get("content", "")).strip()
            turn_role = "model" if role in ("assistant", "model") else "user"
            parts.append(f"{sot}{turn_role}\n{content}{eot}")
        if add_generation_prompt:
            parts.append(f"{sot}model\n")
        return "".join(parts)

    # Fallback: Gemma 2/3 format — <start_of_turn>role\ncontent<end_of_turn>\n
    if not bos:
        for src in (tok, inner):
            if src is None:
                continue
            bos = getattr(src, "bos_token", None) or ""
            if bos:
                break
    bos = bos or "<bos>"
    parts = [bos]
    for msg in messages:
        role    = msg.get("role", "user")
        content = str(msg.get("content", "")).strip()
        turn_role = "model" if role in ("assistant", "model") else "user"
        parts.append(f"<start_of_turn>{turn_role}\n{content}<end_of_turn>\n")
    if add_generation_prompt:
        parts.append("<start_of_turn>model\n")
    return "".join(parts)


def _build_chat_inputs(messages: list, image=None):
    """Build tensors for model.generate(). image is a PIL Image or None."""
    import torch
    device = next(_state._chat_model.parameters()).device
    if _state._chat_is_vision and image is not None:
        # Resolve the correct image placeholder for this processor/tokenizer.
        # Gemma 4 uses <|image|>; older VL models use <image>.
        _img_tok = "<image>"
        for _src in (_state._chat_tok,
                     getattr(_state._chat_tok, "tokenizer", None)):
            if _src is None:
                continue
            _cand = getattr(_src, "image_token", None)
            if _cand:
                _img_tok = _cand
                break
        _user_q = str(messages[-1]["content"])
        if not any(w in _user_q.lower() for w in [
            "<image>", "<|image|>", "image", "picture", "photo", "figure",
            "diagram", "describe", "show", "see", "look", "what is", "what's"
        ]):
            _user_q = f"Please describe what you see in the image. {_user_q}"
        user_text = _img_tok + "\n" + _user_q
        vision_msgs = messages[:-1] + [{"role": "user", "content": user_text}]
        text   = _apply_chat_template(
            _state._chat_tok, vision_msgs, tokenize=False, add_generation_prompt=True)
        inputs = _state._chat_tok(text=text, images=[image], return_tensors="pt",
                                   add_special_tokens=False).to(device)
        inputs.pop("num_patches", None)
        inputs.pop("image_flags", None)
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/vl_debug.log", "a") as _vd2:
                _pv  = inputs.get("pixel_values")
                _ids = inputs.get("input_ids")
                _vd2.write(
                    f"[inputs] keys={list(inputs.keys())} "
                    f"pv={'None' if _pv is None else str(_pv.shape)} "
                    f"ids={'None' if _ids is None else str(_ids.shape)}\n"
                )
                if os.environ.get("VL_DEBUG"):
                    _vd2.write(f"[text] {repr(text[:300])}\n---\n")
        except Exception:
            pass
    else:
        text = _apply_chat_template(
            _state._chat_tok, messages, tokenize=False, add_generation_prompt=True)
        _tok_only = getattr(_state._chat_tok, "tokenizer", _state._chat_tok)
        enc    = _tok_only(text, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in enc.items()}
    return inputs


def _generate_reply(msgs: list, max_tokens: int = 256, image=None) -> str:
    """Unified generation: uses vLLM client if loaded, else transformers."""
    import re as _re

    with _state._vllm_lock:
        _client   = _state._vllm_client
        _model_id = _state._vllm_model_id

    if _client is not None:
        api_msgs = []
        for m in msgs[:-1]:
            api_msgs.append({"role": m["role"], "content": m["content"]})
        last = msgs[-1]
        if image is not None:
            import base64
            from io import BytesIO
            buf  = BytesIO()
            image.save(buf, format="JPEG")
            b64  = base64.b64encode(buf.getvalue()).decode()
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": last["content"]},
            ]
        else:
            content = last["content"]
        api_msgs.append({"role": last["role"], "content": content})
        resp = _client.chat.completions.create(
            model=_model_id,
            messages=api_msgs,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )
        return resp.choices[0].message.content.strip()

    import torch
    torch.cuda.empty_cache()
    inputs = _build_chat_inputs(msgs, image=image)
    ids    = inputs.get("input_ids")
    _prompt_len = ids.shape[1] if ids is not None else 0
    _tokenizer  = getattr(_state._chat_tok, "tokenizer", _state._chat_tok)
    _eos_ids    = [_tokenizer.eos_token_id]
    # Add end-of-turn token as stop token (Gemma 4: <turn|> = 106).
    for _ts in (_state._chat_tok, getattr(_state._chat_tok, "tokenizer", None)):
        _eot_str = getattr(_ts, "eot_token", None)
        if _eot_str:
            try:
                _eot_id = _tokenizer.convert_tokens_to_ids(_eot_str)
                if _eot_id and _eot_id != _tokenizer.eos_token_id:
                    _eos_ids.append(_eot_id)
            except Exception:
                pass
            break
    # Also handle older Qwen/ChatML models that use <|im_end|>.
    try:
        _im_end = _tokenizer.convert_tokens_to_ids("<|im_end|>")
        if _im_end and _im_end != _tokenizer.eos_token_id and _im_end not in _eos_ids:
            _eos_ids.append(_im_end)
    except Exception:
        pass

    with torch.no_grad():
        try:
            if image is not None and _state._chat_is_vision:
                _gen_kwargs = dict(do_sample=False, repetition_penalty=1.4)
            else:
                _gen_kwargs = dict(do_sample=True, temperature=0.7, top_p=0.9,
                                   repetition_penalty=1.3)
            out = _state._chat_model.generate(
                **inputs, max_new_tokens=max_tokens,
                eos_token_id=_eos_ids,
                pad_token_id=_tokenizer.eos_token_id,
                **_gen_kwargs)
        except Exception as _gen_exc:
            import traceback as _tb
            _tb_str = _tb.format_exc()
            try:
                os.makedirs("logs", exist_ok=True)
                with open("logs/ft_chat_errors.log", "a") as _ef:
                    _ef.write(f"[_generate_reply] {type(_gen_exc).__name__}: {_gen_exc}\n{_tb_str}\n---\n")
            except Exception:
                pass
            raise

    # Only decode the newly generated tokens (exclude prompt to avoid echoing history).
    _new_ids  = out[0][_prompt_len:]
    reply_raw = _tokenizer.decode(_new_ids, skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
    try:
        with open("logs/vl_debug.log", "a") as _vd:
            _vd.write(
                f"[gen] prompt_len={_prompt_len} out_len={out[0].shape[0]} "
                f"new_tokens={_new_ids.shape[0]} image={'yes' if image is not None else 'no'}\n"
            )
            _vd.write(f"[raw] {repr(reply_raw[:400])}\n---\n")
    except Exception:
        pass

    reply = _tokenizer.decode(_new_ids, skip_special_tokens=True,
                              clean_up_tokenization_spaces=False).strip()
    reply = _re.sub(r'<0x([0-9A-Fa-f]{2})>',
                    lambda m: bytes([int(m.group(1), 16)]).decode('utf-8', errors='replace'),
                    reply)
    reply = _re.sub(r'<think>.*?</think>', '', reply, flags=_re.DOTALL).strip()
    # Strip residual turn markers from Gemma 4 (<turn|>) or ChatML (<|im_end|>).
    _inner_tok = getattr(_state._chat_tok, "tokenizer", _state._chat_tok)
    _eot_str   = getattr(_inner_tok, "eot_token", None)
    if _eot_str:
        reply = reply.replace(_eot_str, "")
    reply = reply.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    # If the model echoes conversation markers from its training data (e.g. [INST],
    # ### Human:, <|user|>) truncate at the first one — those mark where a new turn
    # would start and everything after is hallucinated dialogue.
    for _turn_marker in ("[INST]", "### Human:", "### User:", "<|user|>",
                         "<|Human|>", "Human:", "User:"):
        _idx = reply.find(_turn_marker)
        if _idx > 0:
            reply = reply[:_idx].strip()
            break
    # Truncate word-soup degeneration: cap at 800 chars and cut at the last clean
    # sentence boundary so the reply ends naturally rather than mid-degeneration.
    _char_limit = 800
    if len(reply) > _char_limit:
        _snip = reply[:_char_limit]
        _last_end = max(_snip.rfind(". "), _snip.rfind("! "), _snip.rfind("? "),
                        _snip.rfind(".\n"), _snip.rfind("!\n"), _snip.rfind("?\n"))
        if _last_end > _char_limit * 0.3:
            reply = reply[:_last_end + 1].strip()
        else:
            reply = _snip.rstrip() + "…"
    return reply
