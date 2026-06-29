# MKD Fine-Tuning Platform — API Specification

> **Last updated:** 2026-06-18
> **Gradio UI (live API explorer):** `https://adfd77dcdc7ed6c00b.gradio.live/?view=api`

Base URL: `http://localhost:8000`
All responses use the envelope:
```json
{ "ok": true,  "data": { ... } }
{ "ok": false, "error": "message", "detail": "..." }
```

## Authentication

Set `API_SECRET` env var on the server. Send `Authorization: Bearer <token>` on every request except `/api/health` and `/api/status`.

---

## System

### `GET /api/health`
No auth required.
**Response:**
```json
{"ok":true,"data":{"status":"ok","ts":"2026-06-16T12:00:00"}}
```

### `GET /api/status`
No auth required.
**Response:**
```json
{
  "ok": true,
  "data": {
    "gpus": [
      {"index":0,"name":"NVIDIA RTX 5090","vram_used_gb":4.2,"vram_total_gb":32.0,"vram_free_gb":27.8,"utilization_pct":12}
    ],
    "disk_free_gb": 480.3,
    "active_job": {
      "run_id":        "run_2026-06-16_0543",
      "step":          420,
      "total_steps":   1200,
      "loss":          0.8231,
      "learning_rate": 1.85e-4,
      "epoch":         1.4,
      "elapsed_seconds": 3600,
      "phase":         "training"
    }
  }
}
```
`active_job` is `null` when no job is running.

---

## Datasets

### `POST /api/datasets/add`
Triggers `ft add-dataset`.
```json
{
  "source":          "tatsu-lab/alpaca",
  "name":            "alpaca",
  "instruction_col": "",
  "response_col":    ""
}
```
**Response:** `{"output": "...terminal output..."}`

### `POST /api/datasets/clean`
Triggers `ft clean-dataset`.
```json
{
  "name":       "alpaca",
  "max_chars":  32768,
  "val_ratio":  0.01,
  "lang":       "en"
}
```

### `GET /api/datasets`
List all datasets.
```json
{
  "ok": true,
  "data": [
    {"name":"alpaca","has_train":true,"train_rows":52000,"val_rows":520,"format":"alpaca","status":"cleaned"}
  ]
}
```

### `GET /api/datasets/{name}/sample`
Returns up to 10 rows from `sample_100.jsonl`.

### `GET /api/datasets/{name}/card`
Returns the full data card JSON.

### `DELETE /api/datasets/{name}`
Removes the dataset directory and data card.

---

## Models

### `POST /api/models/add`
```json
{ "hf_model_id": "google/gemma-4-12b", "local_name": "gemma-4-12b" }
```

### `GET /api/models`
```json
{
  "ok": true,
  "data": [
    {"name":"gemma-4-12b","is_vl":true,"vram_bf16_gb":24.1}
  ]
}
```

### `DELETE /api/models/{name}`

---

## Training

### `POST /api/training/start`
Returns `202 Accepted`.
```json
{
  "model":         "gemma-4-12b",
  "dataset":       "ko-vqa",
  "mode":          "vl",
  "quant":         "bf16",
  "lora_r":        16,
  "lora_alpha":    32,
  "epochs":        3,
  "batch_size":    1,
  "learning_rate": 2e-4,
  "max_steps":     0,
  "wandb_project": "mkd-finetuning",
  "wandb_entity":  "mkd-hika",
  "smoke":         false
}
```
**Response:**
```json
{"ok":true,"data":{"run_id":"run_2026-06-16_0543","status":"started"}}
```

### `POST /api/training/stop`
Terminates the active job.

### `GET /api/training/status`
Poll every 2 seconds for live progress.
```json
{
  "ok": true,
  "data": {
    "running":          true,
    "run_id":           "run_2026-06-16_0543",
    "step":             420,
    "total_steps":      1200,
    "loss":             0.8231,
    "learning_rate":    1.85e-4,
    "epoch":            1.4,
    "elapsed_seconds":  3600,
    "eta_seconds":      2100,
    "gpu_vram_used_gb": 28.4,
    "phase":            "training"
  }
}
```

### `GET /api/training/logs?run_id={id}&offset={n}`
Server-Sent Events stream. Each event:
```
data: {"line": "[step  420] loss=0.8231  lr=1.85e-04", "pos": 420}
```
When the job finishes the stream closes.

---

## Evaluation

### `POST /api/eval/start`
Returns `202 Accepted`.
```json
{
  "checkpoint":    "checkpoints/run_2026-06-16_0543",
  "base_model":    null,
  "suite":         "configs/eval_default.json",
  "skip_mt_bench": true
}
```

### `GET /api/eval/runs`
Returns all runs visible in the Evaluation Dashboard. Every completed training run (Quick Start, Custom Mix, Expert) is **automatically registered** — no manual eval step needed.
```json
{"ok":true,"data":{"runs":["run_2026-06-18_0900","vl_gemma-4-12b_20260618_090512"]}}
```

### `GET /api/eval/results/{run_id}`
```json
{
  "ok": true,
  "data": {
    "run_id":     "run_2026-06-16_0543",
    "summary_md": "# Eval Summary...",
    "harness":    { "mmlu": {...}, "arc_challenge": {...} },
    "vqa":        { "vqa_exact_match": 0.612, "vqa_total": 200, "vqa_correct": 122 }
  }
}
```

### `GET /api/eval/status`
Poll progress of a running evaluation job.
```json
{
  "ok": true,
  "data": {
    "run_id":   "run_2026-06-16_0543",
    "status":   "running",
    "progress": 0.42,
    "message":  "Evaluating VQA samples (21/50)"
  }
}
```
`status` values: `"idle"` | `"running"` | `"done"` | `"failed"`

---

## Export & Publish

### `POST /api/export`
Returns `202 Accepted`.
```json
{ "checkpoint": "checkpoints/run_2026-06-16_0543", "quantize": "none", "merge": true }
```

### `GET /api/export/status`
```json
{"ok":true,"data":{"exports":[{"run":"run_2026-06-16_0543","merged":true}]}}
```

### `POST /api/publish`
`hf_token` is **not** accepted in the request body. Set the `HF_TOKEN` environment variable on the server.
```json
{
  "export_dir":  "exports/run_2026-06-16_0543/merged",
  "hf_repo":     "mkd-hika/gemma-ko-vqa",
  "license":     "apache-2.0",
  "description": "Korean VQA fine-tune of Gemma 4 12B"
}
```

---

## Inference / Chat

> **Image format:** `image_base64` is a plain base64 string — **no** `data:image/...;base64,` prefix. Strip it before sending.

### `POST /api/chat/load`
Load a model into GPU memory.
```json
{ "model_path": "exports/run_2026-06-16_0543/merged" }
```

### `POST /api/chat/unload`
Free GPU memory.

### `POST /api/chat`
```json
{
  "model_path":   "exports/run_2026-06-16_0543/merged",
  "messages": [
    { "role": "user", "content": "What is in this image?" }
  ],
  "image_base64": "<base64 string or null>",
  "max_tokens":   256,
  "stream":       false
}
```
**Non-streaming response:**
```json
{
  "ok": true,
  "data": {
    "role":             "assistant",
    "content":          "The image shows a chart with...",
    "tokens_generated": 42,
    "latency_ms":       810
  }
}
```
**Streaming (`stream: true`):** Returns `text/event-stream`:
```
data: {"token": "The", "done": false}
data: {"token": " image", "done": false}
...
data: {"token": "", "done": true}
```

---

## Error codes

| HTTP | Meaning | Frontend action |
|------|---------|-----------------|
| 200  | Success | Consume `data` |
| 202  | Job accepted (async) | Begin polling the endpoint's `/status` sibling |
| 400  | Bad request / validation error | Show `data.error` to user |
| 401  | Missing `Authorization` header | Prompt user to configure API key |
| 403  | Invalid Bearer token | Show auth error |
| 404  | Resource not found | Show not-found message |
| 422  | Pydantic validation failure | Show field-level error from `detail` array |
| 500  | Server error | Show generic error + suggest checking logs |
