"""
vl_trainer.py — LoRA fine-tuning for multimodal vision-language models (Gemma 4 12B).

Called by app.py as a subprocess:
    python vl_trainer.py --model /path/to/model --dataset /path/to/data.jsonl [options]

Multi-GPU:
    accelerate launch vl_trainer.py --model ... --dataset ...
    (LOCAL_RANK / WORLD_SIZE are set automatically by accelerate/torchrun)

Merge:
    Training saves the LoRA adapter only.
    To merge into a deployable model run:  ft export <output_dir> --merge
"""

import argparse
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Cache pre-population helper
# ---------------------------------------------------------------------------


def _ensure_vl_cache(model_path: str):
    cache_base = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    model_name_san = (
        os.path.basename(os.path.normpath(model_path)).replace("-", "_hyphen_").replace("/", "_")
    )
    cache_model_dir = os.path.join(cache_base, model_name_san)
    if not os.path.isdir(cache_model_dir):
        return
    all_py = glob.glob(os.path.join(model_path, "*.py"))
    for hash_dir in os.listdir(cache_model_dir):
        hash_path = os.path.join(cache_model_dir, hash_dir)
        if not os.path.isdir(hash_path) or hash_dir.startswith("__"):
            continue
        for pyf in all_py:
            dest = os.path.join(hash_path, os.path.basename(pyf))
            if not os.path.exists(dest):
                shutil.copy2(pyf, dest)


# ---------------------------------------------------------------------------
# Loss-mask helper (C-2)
# ---------------------------------------------------------------------------


def _build_labels(tokenizer, msgs: list[dict], input_ids, prompt_fn=None) -> "torch.Tensor":
    """Return labels that supervise only assistant-turn tokens.

    Strategy: use prompt_fn (or apply_chat_template) progressively to find
    where each assistant turn starts/ends in the token sequence.

    prompt_fn: callable(msgs) -> str.  Falls back to tokenizer.apply_chat_template.
    """
    import torch

    labels = torch.full_like(input_ids, -100)

    if not msgs:
        return labels

    def _render(msgs_subset):
        """Render messages to text using prompt_fn or apply_chat_template."""
        if prompt_fn is not None:
            return prompt_fn(msgs_subset)
        return tokenizer.apply_chat_template(
            msgs_subset, tokenize=False, add_generation_prompt=False)

    # Compute BOS offset: does the full encoding prepend a BOS token that the
    # no-special-tokens encoding omits?
    try:
        full_text = _render(msgs)
        no_bos_ids = tokenizer.encode(full_text, add_special_tokens=False)
        bos_offset = max(0, min(len(input_ids) - len(no_bos_ids), 2))
    except Exception:
        bos_offset = 0

    prev_len = 0  # length after previous turn (no-BOS encoding)

    for i, msg in enumerate(msgs):
        try:
            text_i = _render(msgs[: i + 1])
            ids_i = tokenizer.encode(text_i, add_special_tokens=False)
            curr_len = len(ids_i)
        except Exception:
            break

        if msg["role"] == "assistant":
            start = prev_len + bos_offset
            end = min(curr_len + bos_offset, len(labels))
            if start < end:
                labels[start:end] = input_ids[start:end]

        prev_len = curr_len

    return labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class VLDataset:
    """Reads a single JSONL file and returns tokenised examples with loss masks."""

    def __init__(self, jsonl_path: str, processor, max_len: int = 2048):
        self.processor = processor
        self.max_len = max_len
        self.items: list[dict] = []
        _skipped = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        self.items.append(json.loads(line))
                    except json.JSONDecodeError:
                        _skipped += 1
        if _skipped:
            print(f"WARNING: skipped {_skipped} malformed JSONL lines in {jsonl_path}", flush=True)

        # C-3: Pre-flight image validation — raise if failure rate > 5%
        self._validate_images(jsonl_path)

    def _validate_images(self, jsonl_path: str):
        from PIL import Image as _PIL_Image

        total = errors = 0
        for item in self.items:
            img_path = item.get("image")
            if not img_path:
                continue
            total += 1
            if not os.path.exists(img_path):
                errors += 1
            else:
                try:
                    with _PIL_Image.open(img_path) as _im:
                        _im.verify()
                except Exception:
                    errors += 1
        if total > 0 and errors / total > 0.05:
            raise RuntimeError(
                f"Image failure rate {errors}/{total} ({errors/total:.1%}) in "
                f"{jsonl_path} exceeds 5%% threshold. Fix image paths before training."
            )
        if total > 0:
            print(
                f"Image validation: {total - errors}/{total} OK " f"({errors} failures)", flush=True
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        item = self.items[idx]
        convs = item.get("conversations", item.get("messages", []))
        img_path = item.get("image")

        # Normalise role names
        msgs = []
        for c in convs:
            role = c.get("role") or (
                "user" if c.get("from", "") in ("human", "user") else "assistant"
            )
            content = c.get("content") or c.get("value", "")
            msgs.append({"role": role, "content": content})

        # Build prompt with Gemma 4 fallback (no chat_template on processor).
        _inner_tok = getattr(self.processor, "tokenizer", self.processor)
        _sot = getattr(_inner_tok, "sot_token", None)
        _eot = getattr(_inner_tok, "eot_token", None)
        _bos = getattr(_inner_tok, "bos_token", None) or "<bos>"
        _img_tok = getattr(self.processor, "image_token", None) or \
                   getattr(_inner_tok, "image_token", None) or "<image>"

        def _make_prompt(msgs_):
            try:
                return self.processor.apply_chat_template(
                    msgs_, tokenize=False, add_generation_prompt=False)
            except (ValueError, AttributeError):
                pass
            try:
                return _inner_tok.apply_chat_template(
                    msgs_, tokenize=False, add_generation_prompt=False)
            except (ValueError, AttributeError):
                pass
            parts = [_bos]
            for m in msgs_:
                role = "model" if m["role"] in ("assistant", "model") else "user"
                if _sot and _eot:
                    parts.append(f"{_sot}{role}\n{m['content'].strip()}{_eot}")
                else:
                    parts.append(f"<start_of_turn>{role}\n{m['content'].strip()}<end_of_turn>\n")
            return "".join(parts)

        img = None
        if img_path and os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                img = None

        # Inject image token into first user message if image present.
        if img is not None:
            msgs_with_img = []
            _injected = False
            for m in msgs:
                if m["role"] == "user" and not _injected:
                    msgs_with_img.append({"role": "user",
                                          "content": f"{_img_tok}\n{m['content']}"})
                    _injected = True
                else:
                    msgs_with_img.append(m)
            text = _make_prompt(msgs_with_img)
        else:
            text = _make_prompt(msgs)

        if img is not None:
            inputs = self.processor(text=text, images=[img], return_tensors="pt",
                                    add_special_tokens=False)
            inputs.pop("num_patches", None)
        else:
            inputs = _inner_tok(text, return_tensors="pt", add_special_tokens=False)
            inputs.pop("num_patches", None)

        # Squeeze batch dimension
        out: dict = {k: v.squeeze(0) for k, v in inputs.items()}

        # C-2: Build per-sample loss mask — supervise only assistant turns
        out["labels"] = _build_labels(_inner_tok, msgs, out["input_ids"],
                                      prompt_fn=_make_prompt)

        return out


def _load_jsonl_paths(dataset_arg: str) -> list[str]:
    if os.path.isfile(dataset_arg):
        return [dataset_arg]
    if os.path.isdir(dataset_arg):
        # Canonical case: train.jsonl is the training set. val.jsonl is picked
        # up separately for eval by _build_datasets, so it must NOT be globbed
        # in here (otherwise it leaks into training). Intermediate artefacts
        # (normalized.jsonl, sample_100.jsonl) are likewise excluded.
        train_path = os.path.join(dataset_arg, "train.jsonl")
        if os.path.isfile(train_path):
            return [train_path]
        # Fallback for legacy datasets without a train.jsonl: glob everything
        # except known non-training files.
        _excluded = {"val.jsonl", "normalized.jsonl", "sample_100.jsonl"}
        paths = sorted(
            p
            for p in glob.glob(os.path.join(dataset_arg, "*.jsonl"))
            if os.path.basename(p) not in _excluded
        )
        if not paths:
            raise ValueError(f"No .jsonl files found in directory: {dataset_arg}")
        return paths
    raise ValueError(f"--dataset must be a JSONL file or directory: {dataset_arg}")


def _compute_vqa_accuracy(
    model, processor, jsonl_path: str, n_samples: int = 50, device: str = "cuda"
) -> tuple:
    """Compute exact-match VQA accuracy on up to n_samples rows from a JSONL file.
    Returns (accuracy_float, total_evaluated).
    """
    import re as _re
    from pathlib import Path as _Path

    rows = []
    try:
        for line in _Path(jsonl_path).read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return 0.0, 0

    rows = rows[:n_samples]
    correct = 0
    total = 0
    failures = 0

    model.eval()
    import torch as _torch

    with _torch.no_grad():
        for row in rows:
            messages = row.get("messages", [])
            img_paths = row.get("images", [])
            user_q = next((m["content"] for m in messages if m["role"] == "user"), None)
            expected = next((m["content"] for m in messages if m["role"] == "assistant"), None)
            if not user_q or not expected:
                continue

            image = None
            if img_paths:
                try:
                    from PIL import Image as _PILImage

                    image = _PILImage.open(img_paths[0]).convert("RGB")
                except Exception:
                    pass

            try:
                # Resolve image token and build prompt robustly (handles Gemma 4
                # which has no chat_template and uses <|image|> not <image>).
                _inner_tok = getattr(processor, "tokenizer", processor)
                _sot = getattr(_inner_tok, "sot_token", None)
                _eot = getattr(_inner_tok, "eot_token", None)
                _bos = getattr(_inner_tok, "bos_token", None) or "<bos>"
                _img_tok = getattr(processor, "image_token", None) or \
                           getattr(_inner_tok, "image_token", None) or "<image>"

                def _make_prompt(msgs, add_gen=True):
                    """apply_chat_template with Gemma4 fallback."""
                    try:
                        return processor.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=add_gen)
                    except (ValueError, AttributeError):
                        pass
                    try:
                        return _inner_tok.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=add_gen)
                    except (ValueError, AttributeError):
                        pass
                    # Manual fallback
                    parts = [_bos]
                    for m in msgs:
                        role = "model" if m["role"] in ("assistant", "model") else "user"
                        if _sot and _eot:
                            parts.append(f"{_sot}{role}\n{m['content'].strip()}{_eot}")
                        else:
                            parts.append(f"<start_of_turn>{role}\n{m['content'].strip()}<end_of_turn>\n")
                    if add_gen:
                        parts.append(f"{_sot}model\n" if _sot else "<start_of_turn>model\n")
                    return "".join(parts)

                if image is not None:
                    user_text = f"{_img_tok}\n{user_q}"
                    msgs = [{"role": "user", "content": user_text}]
                    text = _make_prompt(msgs)
                    inputs = processor(text=text, images=[image], return_tensors="pt",
                                       add_special_tokens=False).to(device)
                    inputs.pop("image_flags", None)
                    # Flatten dynamic-tiling batch dim: [1, tiles, C, H, W] → [tiles, C, H, W]
                    _pv = inputs.get("pixel_values")
                    if _pv is not None and _pv.dim() == 5:
                        inputs["pixel_values"] = _pv.reshape(-1, *_pv.shape[2:])
                else:
                    msgs = [{"role": "user", "content": user_q}]
                    text = _make_prompt(msgs)
                    inputs = _inner_tok(text, return_tensors="pt",
                                        add_special_tokens=False).to(device)

                n_in = inputs["input_ids"].shape[1]
                _eos_ids = [_inner_tok.eos_token_id]
                if _eot:
                    _eot_id = _inner_tok.convert_tokens_to_ids(_eot)
                    if _eot_id and _eot_id != _inner_tok.eos_token_id:
                        _eos_ids.append(_eot_id)
                out = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    eos_token_id=_eos_ids,
                    pad_token_id=_inner_tok.eos_token_id,
                )
                generated = _inner_tok.decode(
                    out[0, n_in:], skip_special_tokens=True
                ).strip()

                def _norm(s):
                    return _re.sub(r"[^\w\s]", "", s.lower()).strip()

                if _norm(generated) == _norm(expected):
                    correct += 1
                total += 1
            except Exception:
                failures += 1
                continue

    model.train()
    attempted = total + failures
    if attempted > 0 and failures > attempted * 0.5:
        raise RuntimeError(
            f"VQA eval aborted: {failures}/{attempted} samples failed — "
            "check model outputs and dataset format"
        )
    return (correct / total if total > 0 else 0.0), total


def _build_datasets(jsonl_paths: list[str], processor, max_len: int):
    """Build train and eval datasets.

    Returns (train_dataset, eval_dataset_or_None).
    Prefers a paired val.jsonl in the same directory; falls back to 1 % split.
    """
    import torch
    from torch.utils.data import ConcatDataset, Dataset, random_split

    class _TorchVLDataset(Dataset):
        def __init__(self, p, proc, ml):
            self._inner = VLDataset(p, proc, ml)

        def __len__(self):
            return len(self._inner)

        def __getitem__(self, i):
            return self._inner[i]

    train_ds_list: list[Dataset] = []
    eval_ds_list: list[Dataset] = []

    for p in jsonl_paths:
        train_ds_list.append(_TorchVLDataset(p, processor, max_len))
        val_path = Path(p).parent / "val.jsonl"
        if val_path.exists():
            eval_ds_list.append(_TorchVLDataset(str(val_path), processor, max_len))

    train_ds = ConcatDataset(train_ds_list) if len(train_ds_list) > 1 else train_ds_list[0]

    if eval_ds_list:
        eval_ds = ConcatDataset(eval_ds_list) if len(eval_ds_list) > 1 else eval_ds_list[0]
    elif len(train_ds) >= 100:
        n_val = max(1, int(len(train_ds) * 0.01))
        n_train = len(train_ds) - n_val
        train_ds, eval_ds = random_split(
            train_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"No val.jsonl found — auto-split: {n_train} train / {n_val} val", flush=True)
    else:
        eval_ds = None
        print("Dataset too small for 1% split — skipping eval.", flush=True)

    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


def vl_collate_fn(batch: list[dict], processor) -> dict:
    import torch

    pad_id = processor.tokenizer.pad_token_id or 0

    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=pad_id,
    )
    attention_mask = (input_ids != pad_id).long()

    # C-2: Use pre-built labels from the dataset; pad with -100
    if "labels" in batch[0]:
        labels = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=-100,
        )
    else:
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # C-1: Concatenate tiles from ALL image-bearing samples along dim 0.
    # Each sample's pixel_values is [tiles, C, H, W] (variable tile count per
    # image due to dynamic tiling), so they must be CONCATENATED — not stacked
    # — into [total_tiles, C, H, W] (4D). torch.stack would add a batch dim
    # (→ 5D, which the vision encoder rejects) and would fail outright on
    # images with differing tile counts. The model demuxes tiles back to
    # images via the <image> placeholder tokens in input_ids.
    has_img_idx = [i for i, b in enumerate(batch) if "pixel_values" in b]
    if has_img_idx:
        pixel_vals = []
        for i in has_img_idx:
            pv = batch[i]["pixel_values"]
            if pv.dim() == 5:  # [1, tiles, C, H, W] → [tiles, C, H, W]
                pv = pv.reshape(-1, *pv.shape[2:])
            elif pv.dim() == 3:  # [C, H, W] (single tile) → [1, C, H, W]
                pv = pv.unsqueeze(0)
            pixel_vals.append(pv)
        out["pixel_values"] = torch.cat(pixel_vals, dim=0)

        # image_flags must have ONE entry per image TILE (matching dim 0 of the
        # vision embeddings the model produces from pixel_values), not one per
        # sample. The model does `vit_embeds[image_flags == 1]`, so a per-sample
        # mask of length B mismatches the [total_tiles, ...] embedding tensor.
        total_tiles = out["pixel_values"].shape[0]
        out["image_flags"] = torch.ones(total_tiles, 1, dtype=torch.long)

    return out


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class StreamCallback:
    """Streams step-level metrics to stdout for app.py to parse."""

    _TrainerCallback = None

    def __new__(cls, *args, **kwargs):
        from transformers import TrainerCallback

        if cls._TrainerCallback is None:
            cls._TrainerCallback = TrainerCallback

        class _Impl(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                print(f"TOTAL_STEPS={state.max_steps}", flush=True)

            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    step = state.global_step
                    loss = logs["loss"]
                    lr = logs.get("learning_rate", 0)
                    epoch = logs.get("epoch", state.epoch or 0)
                    print(
                        f"[step {step:4d}] loss={loss:.4f}  lr={lr:.2e}  epoch={epoch:.2f}",
                        flush=True,
                    )

        return _Impl()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for multimodal vision-language models")
    parser.add_argument("--model", required=True, help="Local model path")
    parser.add_argument("--dataset", required=True, help="JSONL file or directory of JSONL files")
    parser.add_argument(
        "--output", default=None, help="Output dir for LoRA adapter (default: checkpoints/vl_<ts>)"
    )
    parser.add_argument("--quant", choices=["bf16", "8bit", "4bit"], default="bf16")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=0, help="If >0, overrides --epochs")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Per-device train batch size (default 1). " "Batch 2 fits 32 GB with 4-bit QLoRA.",
    )
    parser.add_argument(
        "--max-image-tiles",
        type=int,
        default=4,
        help="Cap dynamic image tiling (model default 12). "
        "Fewer tiles = shorter sequence = less memory. "
        "4 keeps a 12B VL model in bf16 under 32 GB.",
    )
    parser.add_argument(
        "--wandb-project", default="", help="W&B project name (empty = disable W&B)"
    )
    parser.add_argument("--wandb-entity", default="", help="W&B entity / team name")
    parser.add_argument("--smoke", action="store_true", help="Quick validation: run only 5 steps")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Disable transformers 5.x threaded model loading — it uses ThreadPoolExecutor
    # which causes CUDA driver initialization errors on some NVIDIA environments.
    os.environ["HF_ENABLE_PARALLEL_LOADING"] = "0"

    args = parse_args()

    # LOW: Explicit path validation at the start of main()
    if not os.path.isdir(args.model):
        print(f"ERROR: --model path does not exist or is not a directory: {args.model}", flush=True)
        sys.exit(1)
    # Dataset existence is validated by _load_jsonl_paths below

    # Resolve output directory
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join("checkpoints", f"vl_{ts}")
    os.makedirs(args.output, exist_ok=True)

    print(f"Output directory : {args.output}", flush=True)
    print(f"Model            : {args.model}", flush=True)
    print(f"Dataset          : {args.dataset}", flush=True)
    print(f"Quantisation     : {args.quant}", flush=True)
    print(f"LoRA r/alpha     : {args.lora_r}/{args.lora_alpha}", flush=True)
    if args.smoke:
        print("Smoke mode       : ON (5 steps)", flush=True)

    # M-3: W&B setup
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        run_name = f"gemma-vl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_to = "wandb"
    else:
        run_name = None
        report_to = "none"

    # Deferred heavy imports
    from functools import partial

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoProcessor

    # H-1: Determine device placement based on GPU environment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    n_gpus = torch.cuda.device_count()

    if local_rank >= 0:
        # Running under torchrun / accelerate launch
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
        print(f"Distributed mode: LOCAL_RANK={local_rank}", flush=True)
    elif n_gpus > 1:
        # Multi-GPU without torchrun: load on GPU 0 only.
        # For true data parallelism, launch with torchrun --nproc_per_node=N
        device_map = {"": "cuda:0"}
        print(f"Multi-GPU detected ({n_gpus} GPUs) but no torchrun — using GPU 0 only. "
              f"Use torchrun for data parallelism.", flush=True)
    else:
        device_map = {"": "cuda:0"}
        print('Single GPU: device_map={"":"cuda:0"}', flush=True)

    _ensure_vl_cache(args.model)

    # Load processor
    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    # Cap dynamic image tiling. The model defaults to up to 12 tiles per image,
    # which produces very long sequences (≈256 tokens/tile) and a cross-entropy
    # logits spike that OOMs a 12B bf16 model on a 32 GB card. Capping the tile
    # count shrinks the sequence enough to fit. Applies to both the training
    # dataset and the VQA eval, since they share this processor instance.
    if hasattr(processor, "image_processor") and hasattr(
        processor.image_processor, "max_num_tiles"
    ):
        _was = processor.image_processor.max_num_tiles
        processor.image_processor.max_num_tiles = args.max_image_tiles
        print(f"Capped image tiles at {args.max_image_tiles} (was {_was}).", flush=True)

    # Build load kwargs
    load_kwargs: dict = dict(
        device_map=device_map,
        trust_remote_code=True,
    )

    # M-2: Flash Attention with graceful fallback
    try:
        import flash_attn  # noqa: F401

        load_kwargs["attn_implementation"] = "flash_attention_2"
        print("Using FlashAttention-2.", flush=True)
    except Exception:
        # SDPA with cuDNN backend can fail on some environments.
        # Disable cuDNN SDPA, let PyTorch use flash_sdp or math fallback.
        torch.backends.cuda.enable_cudnn_sdp(False)
        load_kwargs["attn_implementation"] = "sdpa"
        print("Using PyTorch SDPA (cuDNN disabled, flash_sdp/math).", flush=True)

    if args.quant == "8bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant == "4bit":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # Always load the non-quantized layers (embeddings, norms, vision tower) in
    # bf16. Without this on the 4/8-bit paths they default to fp32, and merging
    # the bf16 image features into the fp32 text embeddings fails with an
    # "Index put dtype mismatch" error in the VL forward.
    load_kwargs["dtype"] = torch.bfloat16

    # Load model
    # Disable transformers 5.x threaded weight loading — ThreadPoolExecutor
    # causes CUDA driver initialization errors on NVIDIA cloud environments.
    try:
        import transformers.core_model_loading as _cml
        _cml.GLOBAL_WORKERS = 1
    except Exception:
        pass
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    # QLoRA: a quantized base model must be prepped before LoRA is attached.
    # prepare_model_for_kbit_training casts layernorms to fp32, enables input
    # gradients, and wires up the dequantization path needed for the backward
    # pass. Without it, 4/8-bit weights fail with a packed-weight matmul error
    # during training (inference still works, which masks the bug).
    if args.quant in ("4bit", "8bit"):
        # The top-level VL wrapper doesn't implement
        # gradient_checkpointing_enable, so tell prepare_* to skip it
        # (use_gradient_checkpointing=False) — GC is enabled on the inner
        # language_model further below. We still must enable input grads so
        # gradients propagate through the frozen quantized base when GC is on.
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print(f"Prepared {args.quant} model for k-bit training.", flush=True)

    # LoRA
    print("Applying LoRA...", flush=True)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            # Attention projections (Gemma attention layers)
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # Gate/value projections
            "in_proj",
            "out_proj",
            # MLP projections (SwiGLU feed-forward blocks)
            "up_proj",
            "down_proj",
            # NOTE: Vision encoder layers (modeling.py mlp1 / mlp2) use unnamed
            # nn.Sequential — not directly targetable by module name. To fine-tune
            # the vision-language projection, add vision_projection training via
            # a custom trainer or unfreeze mlp1/mlp2 manually.
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Gradient checkpointing on inner language_model (for VL wrappers)
    try:
        model.base_model.model.language_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on language_model.", flush=True)
    except Exception as e:
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.", flush=True)
        except Exception:
            print(f"Gradient checkpointing skipped: {e}", flush=True)

    # Dataset
    jsonl_paths = _load_jsonl_paths(args.dataset)
    print(f"JSONL files found: {len(jsonl_paths)}", flush=True)
    train_dataset, eval_dataset = _build_datasets(jsonl_paths, processor, max_len=2048)
    print(f"Train samples: {len(train_dataset)}", flush=True)
    if eval_dataset is not None:
        print(f"Eval  samples: {len(eval_dataset)}", flush=True)

    # Training arguments
    from transformers import Trainer, TrainingArguments

    _n_workers = 0 if args.smoke else 4
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=1 if args.smoke else args.epochs,
        max_steps=5 if args.smoke else (args.max_steps if args.max_steps > 0 else -1),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=1,
        # H-3: Checkpoint management
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        # H-2: Evaluation. eval_steps must divide save_steps evenly when
        # load_best_model_at_end is set (HF Trainer requirement), so keep them
        # aligned at 500.
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=500,
        load_best_model_at_end=(eval_dataset is not None),
        # Only compute eval LOSS — do not gather/accumulate the full logits
        # tensor across eval samples. With a large vocab and long VL sequences,
        # logit accumulation OOMs (4+ GiB) even when training itself fits.
        prediction_loss_only=True,
        per_device_eval_batch_size=1,
        # M-1: Dataloader performance
        dataloader_num_workers=_n_workers,
        dataloader_pin_memory=(_n_workers > 0),
        dataloader_persistent_workers=(_n_workers > 0),
        # M-6: Resume from latest checkpoint if one exists
        resume_from_checkpoint=True,
        # M-3: Reporting
        report_to=report_to,
        run_name=run_name,
        # LOW: Reproducibility
        seed=42,
        remove_unused_columns=False,
        # LoRA freezes most parameters — DDP must not error on unused params
        ddp_find_unused_parameters=False,
    )

    collator = partial(vl_collate_fn, processor=processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[StreamCallback()],
    )

    # Find a val.jsonl for VQA accuracy evaluation
    _vqa_val_path = None
    for _jp in jsonl_paths:
        _candidate = Path(_jp).parent / "val.jsonl"
        if _candidate.exists():
            _vqa_val_path = str(_candidate)
            break

    # Compute baseline VQA accuracy (before training, LoRA weights ≈ 0)
    _vqa_before, _vqa_n = 0.0, 0
    if _vqa_val_path:
        print("Computing baseline VQA accuracy (before training)...", flush=True)
        _vqa_before, _vqa_n = _compute_vqa_accuracy(
            model, processor, _vqa_val_path, n_samples=50, device=device
        )
        print(f"Baseline VQA accuracy: {_vqa_before:.1%} on {_vqa_n} samples", flush=True)

    # Free the generation KV-cache and fragmentation left by the baseline VQA
    # generate() pass before training begins. Without this, that residual
    # allocation tips a 12B bf16 model over the 32 GB limit on the first
    # optimizer step (the cross-entropy logits spike needs the headroom).
    import gc as _gc

    import torch as _torch_cleanup

    _gc.collect()
    if _torch_cleanup.cuda.is_available():
        _torch_cleanup.cuda.empty_cache()

    # Train
    print("Starting training...", flush=True)
    trainer.train()

    # Compute fine-tuned VQA accuracy (after training)
    _vqa_after = 0.0
    if _vqa_val_path:
        print("Computing fine-tuned VQA accuracy (after training)...", flush=True)
        _vqa_after, _vqa_n = _compute_vqa_accuracy(
            model, processor, _vqa_val_path, n_samples=50, device=device
        )
        print(f"Fine-tuned VQA accuracy: {_vqa_after:.1%} on {_vqa_n} samples", flush=True)
        _vqa_metrics = {
            "vqa_accuracy_before": round(_vqa_before, 4),
            "vqa_accuracy_after": round(_vqa_after, 4),
            "vqa_n_samples": _vqa_n,
        }
        # Write metrics file so the eval dashboard can read it
        with open(Path(args.output) / "vqa_metrics.json", "w") as _mf:
            json.dump(_vqa_metrics, _mf)
        # Output parseable line for app.py
        print(f"VQA_METRICS={json.dumps(_vqa_metrics)}", flush=True)

    # Save adapter only for real training runs — skip for smoke tests
    if args.smoke:
        print("Smoke test complete — skipping adapter save.", flush=True)
        print(f"EXPORT_PATH={args.output}", flush=True)
    else:
        print("Saving LoRA adapter and processor...", flush=True)
        model.save_pretrained(args.output)
        processor.save_pretrained(args.output)

        # Copy custom model .py files so trust_remote_code works from the adapter dir
        for py_file in Path(args.model).glob("*.py"):
            shutil.copy2(str(py_file), str(Path(args.output) / py_file.name))

        print(f"Adapter saved to: {args.output}", flush=True)
        print("To merge LoRA into base model for deployment run:", flush=True)
        print(f"    ft export {args.output} --merge", flush=True)
        print(f"EXPORT_PATH={args.output}", flush=True)

    try:
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        print("GPU memory cleared.", flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
