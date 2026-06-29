# MKD Fine-Tuning Platform — Frontend Contract

> **Last updated:** 2026-06-18
> **Gradio live demo + API explorer:** `https://adfd77dcdc7ed6c00b.gradio.live/?view=api`
> **OpenAPI schema:** `GET /api/openapi.json` | **Swagger UI:** `/api/docs`

This document is the single source of truth for the frontend team.
The backend API is at `http://localhost:8000` (FastAPI, `api.py`).
The Gradio UI is at `http://localhost:7860` (for internal use).

---

## Base URL

```
http://localhost:8000
```

For production replace with your deployed domain.

---

## Auth

Set `API_SECRET` on the server. Every request (except health/status) needs:

```
Authorization: Bearer <API_SECRET>
```

Example with fetch:
```js
const headers = {
  "Content-Type": "application/json",
  "Authorization": `Bearer ${import.meta.env.VITE_API_SECRET}`,
};
```

No auth needed: `GET /api/health`, `GET /api/status`.

---

## Response envelope

Every response (success or error) is:
```json
{ "ok": true,  "data": { ... } }
{ "ok": false, "error": "human-readable message", "detail": "..." }
```

Always check `response.ok && body.ok` before using `body.data`.

---

## Endpoints

### Health check
```js
const res = await fetch("/api/health");
const body = await res.json();
// body.data.status === "ok"
```

### System status (GPU + active job)
```js
const { data } = await (await fetch("/api/status", { headers })).json();
// data.gpus[0].vram_free_gb, data.active_job
```

### Add dataset
```js
await fetch("/api/datasets/add", {
  method: "POST", headers,
  body: JSON.stringify({ source: "tatsu-lab/alpaca", name: "alpaca" }),
});
```

### List datasets
```js
const { data } = await (await fetch("/api/datasets", { headers })).json();
// data is an array of { name, status, train_rows, val_rows, format, ... }
```

### Preview 10 rows
```js
const { data } = await (await fetch(`/api/datasets/${name}/sample`, { headers })).json();
// data.rows — array of ChatML objects
```

### Start training (VL mode)
```js
const { data } = await (await fetch("/api/training/start", {
  method: "POST", headers,
  body: JSON.stringify({
    model:         "gemma-4-12b",
    dataset:       "ko-vqa",
    mode:          "vl",
    quant:         "bf16",
    lora_r:        16,
    lora_alpha:    32,
    epochs:        3,
    batch_size:    1,
    learning_rate: 2e-4,
    wandb_project: "mkd-finetuning",
    wandb_entity:  "mkd-hika",
  }),
})).json();
const runId = data.run_id;
```

### Poll training progress (every 2 s)
```js
const poll = setInterval(async () => {
  const { data } = await (await fetch("/api/training/status", { headers })).json();
  if (!data.running) { clearInterval(poll); return; }
  setProgress({
    step:    data.step,
    loss:    data.loss,
    epoch:   data.epoch,
    elapsed: data.elapsed_seconds,
  });
}, 2000);
```

---

## SSE: Live training logs

Use the browser's `EventSource` API (no auth header support) — use `fetch` with
`ReadableStream` instead so you can send the `Authorization` header:

```js
async function* streamLogs(runId, offset = 0) {
  const res = await fetch(
    `/api/training/logs?run_id=${runId}&offset=${offset}`,
    { headers }
  );
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const events = buf.split("\n\n");
    buf = events.pop() ?? "";
    for (const evt of events) {
      const dataLine = evt.split("\n").find(l => l.startsWith("data: "));
      if (dataLine) yield JSON.parse(dataLine.slice(6));
    }
  }
}

// Usage:
for await (const { line } of streamLogs(runId)) {
  appendToTerminal(line);
}
```

---

## SSE: Streaming chat

```js
async function* streamChat(messages, imageBase64 = null) {
  const res = await fetch("/api/chat", {
    method: "POST", headers,
    body: JSON.stringify({ messages, image_base64: imageBase64,
                           max_tokens: 256, stream: true }),
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const events = buf.split("\n\n");
    buf = events.pop() ?? "";
    for (const evt of events) {
      const dataLine = evt.split("\n").find(l => l.startsWith("data: "));
      if (dataLine) {
        const { token, done: finished } = JSON.parse(dataLine.slice(6));
        if (finished) return;
        yield token;
      }
    }
  }
}

// Usage:
let reply = "";
for await (const token of streamChat(messages, imageBase64)) {
  reply += token;
  setChatReply(reply);  // update UI incrementally
}
```

---

## Image upload for multimodal chat

The API accepts `image_base64` — a plain base64 string, **no** `data:image/...;base64,` prefix.
`FileReader.readAsDataURL` adds that prefix — strip it before sending (`.split(",")[1]`).

```js
async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => resolve(reader.result.split(",")[1]); // strip data: prefix
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

const base64 = await fileToBase64(imageFile);  // imageFile from <input type="file">
```

Supported formats: JPEG, PNG, WEBP. Max recommended size: 5 MB.

---

## Training progress display

Recommended polling interval: **2 seconds**.

Fields to display:
| Field | Display |
|---|---|
| `step / total_steps` | Progress bar (hide until `total_steps > 0`) |
| `loss` | Line chart, update each poll |
| `learning_rate` | Badge |
| `epoch` | Badge (e.g. "Epoch 1.4 / 3") |
| `elapsed_seconds` | Timer, format as hh:mm:ss |
| `phase` | Status chip (starting / training / done) |
| `gpu_vram_used_gb` | VRAM gauge |

---

## End-to-end flow

> **Eval auto-registration:** Every completed training run (Quick Start, Custom Mix, Expert — both text and VL) is automatically registered in the Evaluation Dashboard. No separate eval trigger is required to populate the dashboard. Call `GET /api/eval/results/{run_id}` immediately after training completes.

```
1. Add dataset:    POST /api/datasets/add   { source: "...", name: "ko-vqa" }
2. Clean dataset:  POST /api/datasets/clean { name: "ko-vqa", lang: "ko,en" }
3. Preview:        GET  /api/datasets/ko-vqa/sample
4. Start training: POST /api/training/start { model: "...", dataset: "ko-vqa", ... }
   ↳ poll:         GET  /api/training/status  (every 2 s)
   ↳ stream logs:  GET  /api/training/logs?run_id=run_...
5. Evaluate:       POST /api/eval/start      { checkpoint: "checkpoints/run_..." }
   ↳ results:      GET  /api/eval/results/run_...
6. Export:         POST /api/export          { checkpoint: "...", merge: true }
7. Load model:     POST /api/chat/load       { model_path: "exports/run_.../merged" }
8. Chat:           POST /api/chat            { messages: [...], image_base64: "..." }
9. Publish:        POST /api/publish         { export_dir: "...", hf_repo: "..." }
   ↳ status:       GET  /api/eval/status     (poll running eval job)
```

---

## Error handling

| `ok` | HTTP | What to show |
|------|------|---|
| `false` | 400 | Toast: `body.error` |
| `false` | 401 | Redirect to settings page (configure API key) |
| `false` | 403 | Toast: "Invalid API key" |
| `false` | 404 | Inline: "Not found — did training complete?" |
| `false` | 202 | Not an error — job started, begin polling |
| `false` | 500 | Toast: "Server error" + "Check server logs" |

---

## OpenAPI schema

```
GET /api/openapi.json
```

FastAPI auto-generates this. Import into Postman or use Swagger UI at `/api/docs`.
