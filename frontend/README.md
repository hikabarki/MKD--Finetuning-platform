# MKD Fine-Tuning Platform — Frontend Integration

Everything the frontend team needs is in this folder.

---

## Quick Start

| Step | What to read |
|------|-------------|
| 1. Understand the API | `API_SPEC.md` — all endpoints, request/response shapes, error codes |
| 2. Start integrating | `FRONTEND_CONTRACT.md` — copy-paste JS fetch examples for every endpoint |
| 3. Test live | Open `http://localhost:8000/api/docs` — Swagger UI, test any endpoint in the browser |

---

## Base URL

```
http://localhost:8000
```

Replace with your deployed domain in production.

---

## Auth

Every request (except `/api/health` and `/api/status`) needs:

```js
headers: {
  "Content-Type": "application/json",
  "Authorization": `Bearer ${import.meta.env.VITE_API_SECRET}`
}
```

---

## Files in this folder

| File | Purpose |
|------|---------|
| `FRONTEND_CONTRACT.md` | **Start here** — JS fetch examples, SSE streaming, image upload, end-to-end flow |
| `API_SPEC.md` | Full REST API reference — all endpoints, fields, response shapes, error codes |
| `openapi.json` | Auto-generated OpenAPI schema — import into Postman or use with codegen tools |

---

## Live Resources

| Resource | URL |
|----------|-----|
| Swagger UI (interactive) | `http://localhost:8000/api/docs` |
| OpenAPI JSON schema | `http://localhost:8000/api/openapi.json` |
| Gradio UI (internal) | `http://localhost:7860` |

---

## Key Rules

1. **Always check `body.ok`** before using `body.data` — every response is `{ ok, data }` or `{ ok, error }`
2. **NVFP4 is inference-only** — do not show it in training model selectors
3. **Image base64** — strip the `data:image/...;base64,` prefix before sending (use `.split(",")[1]`)
4. **`HF_TOKEN` is server-only** — never ask the user for it in the frontend; it is set as a server env var
5. **Training is async** — `POST /api/training/start` returns `202`, then poll `GET /api/training/status` every 2 seconds
