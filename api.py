"""
api.py — FastAPI REST layer for the MKD Fine-Tuning Platform.

Run alongside Gradio:
    Gradio: python app.py --port 7860
    API:    uvicorn api:app --port 8000 --reload

Or mount Gradio into FastAPI on the same port (see bottom of file).

Auth: set API_SECRET env var.  All endpoints except /api/health and /api/status
require  Authorization: Bearer <API_SECRET>  when API_SECRET is set.

CORS: accepts http://localhost:3000 and http://localhost:5173 by default.
Override with CORS_ORIGINS env var (comma-separated).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# ── Project root ──────────────────────────────────────────────────────────────

ROOT    = Path(os.environ.get("FT_PROJECT", Path(__file__).resolve().parent))
SCRIPTS = ROOT / "scripts"
PYTHON  = sys.executable

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MKD Fine-Tuning Platform API",
    description="REST API for dataset management, training, evaluation, export, and inference.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────

_cors_origins_env = os.environ.get("CORS_ORIGINS", "")
CORS_ORIGINS = (
    [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins_env
    else ["http://localhost:3000", "http://localhost:5173"]
)

_allowed_hosts_env = os.environ.get("ALLOWED_HOSTS", "")
ALLOWED_HOSTS = (
    [h.strip() for h in _allowed_hosts_env.split(",") if h.strip()]
    if _allowed_hosts_env
    else ["localhost", "127.0.0.1", "*.mkd.kr"]
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────

_API_SECRET = os.environ.get("API_SECRET", "").strip()

_PUBLIC_PATHS = {"/api/health", "/api/status"}


def require_auth(request: Request):
    if not _API_SECRET:
        return  # auth disabled
    if request.url.path in _PUBLIC_PATHS:
        return  # no auth for public endpoints
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    if token != _API_SECRET:
        raise HTTPException(status_code=403, detail="Invalid API token")


# ── Response helpers ──────────────────────────────────────────────────────────

def _safe_name(s: str) -> str:
    """Reject names that could escape the project root via path traversal."""
    if not s or "/" in s or ".." in s or "\x00" in s:
        raise HTTPException(status_code=400, detail=f"Invalid name: {s!r}")
    return s


def ok_resp(data: Any) -> dict:
    return {"ok": True, "data": data}


def err_resp(message: str, detail: str = "") -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"ok": False, "error": message, "detail": detail},
    )


# ── In-process job tracker ────────────────────────────────────────────────────

class _Job:
    def __init__(self, run_id: str, proc: subprocess.Popen, log_path: Path):
        self.run_id   = run_id
        self.proc     = proc
        self.log_path = log_path
        self.step     = 0
        self.total_steps = 0
        self.loss     = 0.0
        self.lr       = 0.0
        self.epoch    = 0.0
        self.phase    = "starting"
        self.start_ts = time.time()
        self._thread: threading.Thread | None = None

    def start_monitor(self):
        """Background thread that tails the process stdout into log_path."""
        def _tail():
            with self.log_path.open("w") as lf:
                for raw in self.proc.stdout:  # type: ignore[union-attr]
                    line = raw.rstrip()
                    lf.write(line + "\n")
                    lf.flush()
                    # Parse total steps emitted at train start:
                    # "TOTAL_STEPS=1200"
                    if line.startswith("TOTAL_STEPS="):
                        try:
                            self.total_steps = int(line.split("=", 1)[1])
                        except Exception:
                            pass
                    # Parse step/loss from StreamCallback format:
                    # "[step  420] loss=0.8231  lr=1.85e-04  epoch=0.35"
                    elif line.startswith("[step"):
                        parts = line.split()
                        try:
                            self.step = int(parts[1].rstrip("]"))
                            self.loss = float(parts[2].split("=")[1])
                            self.lr   = float(parts[3].split("=")[1])
                            self.phase = "training"
                        except Exception:
                            pass
                        # Parse epoch if present
                        for p in parts:
                            if p.startswith("epoch="):
                                try:
                                    self.epoch = float(p.split("=")[1])
                                except Exception:
                                    pass
                    if "EXPORT_PATH=" in line:
                        self.phase = "done"
            self.proc.wait()

        self._thread = threading.Thread(target=_tail, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self.proc.poll() is None

    def elapsed(self) -> float:
        return time.time() - self.start_ts


_active_job: _Job | None = None
_active_job_lock = threading.Lock()

# Lightweight tracker for background eval/export jobs
class _BgJob:
    def __init__(self, run_id: str, kind: str, proc: subprocess.Popen, log_path: Path):
        self.run_id   = run_id
        self.kind     = kind  # "eval" or "export"
        self.proc     = proc
        self.log_path = log_path
        self.start_ts = time.time()
        self._thread: threading.Thread | None = None

    def start_tail(self):
        def _tail():
            with self.log_path.open("w") as lf:
                for raw in self.proc.stdout:  # type: ignore[union-attr]
                    lf.write(raw); lf.flush()
            self.proc.wait()
        self._thread = threading.Thread(target=_tail, daemon=True)
        self._thread.start()

    def is_running(self) -> bool:
        return self.proc.poll() is None

    def elapsed(self) -> float:
        return time.time() - self.start_ts

    def status_dict(self) -> dict:
        return {
            "run_id":          self.run_id,
            "kind":            self.kind,
            "running":         self.is_running(),
            "exit_code":       self.proc.returncode,
            "elapsed_seconds": round(self.elapsed()),
            "log":             str(self.log_path),
        }

_bg_jobs: dict[str, _BgJob] = {}
_bg_jobs_lock = threading.Lock()

# ── Helper utilities ──────────────────────────────────────────────────────────

def _list_datasets() -> list[dict]:
    d = ROOT / "datasets"
    if not d.exists():
        return []
    result = []
    for p in sorted(d.iterdir()):
        if not p.is_dir() or p.name.startswith("."):
            continue
        card_file = ROOT / "data_cards" / f"{p.name}.json"
        card = {}
        if card_file.exists():
            try:
                card = json.loads(card_file.read_text())
            except Exception:
                pass
        result.append({
            "name":        p.name,
            "has_raw":     (p / "raw").exists(),
            "has_train":   (p / "train.jsonl").exists(),
            "has_val":     (p / "val.jsonl").exists(),
            "train_rows":  card.get("train_rows"),
            "val_rows":    card.get("val_rows"),
            "format":      card.get("detected_format"),
            "status":      "cleaned" if (p / "train.jsonl").exists() else "raw",
        })
    return result


def _list_models() -> list[dict]:
    d = ROOT / "models"
    if not d.exists():
        return []
    result = []
    for p in sorted(d.iterdir()):
        if not p.is_dir():
            continue
        cfg_path = p / "config.json"
        is_vl = False
        vram_gb = None
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text())
                is_vl = "vision_config" in cfg or any(
                    "vl" in str(v).lower() or "vision" in str(v).lower()
                    for v in cfg.get("auto_map", {}).values()
                )
                params = cfg.get("num_parameters") or cfg.get("hidden_size", 4096)
                if isinstance(params, int) and params > 10_000:
                    # very rough: 2 bytes per param for BF16
                    vram_gb = round(params * 2 / 1e9, 1)
            except Exception:
                pass
        result.append({
            "name":    p.name,
            "is_vl":   is_vl,
            "vram_bf16_gb": vram_gb,
        })
    return result


def _get_gpu_info() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=4,
        ).strip().splitlines()
        def _si(v: str, default: int = 0) -> int:
            try:
                return int(v)
            except (ValueError, TypeError):
                return default

        gpus = []
        for row in out:
            idx, name, used, total, util = [x.strip() for x in row.split(",")]
            used_i  = _si(used)
            total_i = _si(total)
            gpus.append({
                "index":           _si(idx),
                "name":            name,
                "vram_used_gb":    round(used_i / 1024, 2),
                "vram_total_gb":   round(total_i / 1024, 2),
                "vram_free_gb":    round((total_i - used_i) / 1024, 2),
                "utilization_pct": _si(util),
            })
        return gpus
    except Exception:
        return []


def _disk_free_gb(path: Path = ROOT) -> float:
    import shutil as _sh
    st = _sh.disk_usage(str(path))
    return round(st.free / 1e9, 1)



# ═════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════════

# ── System ────────────────────────────────────────────────────────────────────

@app.get("/api/health", dependencies=[Depends(require_auth)])
def health():
    return ok_resp({"status": "ok", "ts": datetime.utcnow().isoformat()})


@app.get("/api/status", dependencies=[Depends(require_auth)])
def api_status():
    with _active_job_lock:
        job = _active_job
    active = None
    if job and job.is_running():
        active = {
            "run_id":         job.run_id,
            "step":           job.step,
            "total_steps":    job.total_steps,
            "loss":           job.loss,
            "learning_rate":  job.lr,
            "epoch":          job.epoch,
            "elapsed_seconds": round(job.elapsed()),
            "phase":          job.phase,
        }
    return ok_resp({
        "gpus":        _get_gpu_info(),
        "disk_free_gb": _disk_free_gb(),
        "active_job":  active,
        "models_loaded": [],  # populated if in-process inference is running
    })


# ── Datasets ──────────────────────────────────────────────────────────────────

class DatasetAddRequest(BaseModel):
    source:          str           = Field(..., max_length=500)
    name:            str | None    = Field(None, max_length=100)
    instruction_col: str           = Field("",   max_length=100)
    response_col:    str           = Field("",   max_length=100)


@app.post("/api/datasets/add", dependencies=[Depends(require_auth)])
def dataset_add(req: DatasetAddRequest):
    cmd = [PYTHON, str(SCRIPTS / "ft_add_dataset.py"), req.source]
    if req.name:
        _safe_name(req.name)
        cmd += ["--name", req.name]
    if req.instruction_col:
        cmd += ["--instruction-col", req.instruction_col]
    if req.response_col:
        cmd += ["--response-col", req.response_col]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=3600)
    if result.returncode != 0:
        return err_resp("ft add-dataset failed", result.stderr[-1000:])
    return ok_resp({"output": result.stdout[-2000:]})


class DatasetCleanRequest(BaseModel):
    name:      str   = Field(...,   max_length=100)
    max_chars: int   = Field(32768, ge=1, le=2_000_000)
    val_ratio: float = Field(0.01,  ge=0.0, le=1.0)
    lang:      str   = Field("",   max_length=10)


@app.post("/api/datasets/clean", dependencies=[Depends(require_auth)])
def dataset_clean(req: DatasetCleanRequest):
    _safe_name(req.name)
    cmd = [PYTHON, str(SCRIPTS / "ft_clean_dataset.py"), req.name,
           f"--max-chars={req.max_chars}", f"--val-ratio={req.val_ratio}"]
    if req.lang:
        cmd += [f"--lang={req.lang}"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=3600)
    if result.returncode != 0:
        return err_resp("ft clean-dataset failed", result.stderr[-1000:])
    return ok_resp({"output": result.stdout[-2000:]})


@app.get("/api/datasets", dependencies=[Depends(require_auth)])
def datasets_list():
    return ok_resp(_list_datasets())


@app.get("/api/datasets/{name}/sample", dependencies=[Depends(require_auth)])
def dataset_sample(name: str):
    _safe_name(name)
    p = ROOT / "datasets" / name / "sample_100.jsonl"
    if not p.exists():
        raise HTTPException(404, f"No sample found for dataset '{name}'")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return ok_resp({"rows": rows[:10]})


@app.get("/api/datasets/{name}/card", dependencies=[Depends(require_auth)])
def dataset_card(name: str):
    _safe_name(name)
    p = ROOT / "data_cards" / f"{name}.json"
    if not p.exists():
        raise HTTPException(404, f"No data card for '{name}'")
    return ok_resp(json.loads(p.read_text()))


@app.delete("/api/datasets/{name}", dependencies=[Depends(require_auth)])
def dataset_delete(name: str):
    _safe_name(name)
    import shutil as _sh
    ds_dir = ROOT / "datasets" / name
    if not ds_dir.exists():
        raise HTTPException(404, f"Dataset '{name}' not found")
    _sh.rmtree(str(ds_dir))
    card = ROOT / "data_cards" / f"{name}.json"
    if card.exists():
        card.unlink()
    return ok_resp({"deleted": name})


# ── Models ────────────────────────────────────────────────────────────────────

class ModelAddRequest(BaseModel):
    hf_model_id: str        = Field(...,  max_length=200)
    local_name:  str | None = Field(None, max_length=100)


@app.post("/api/models/add", dependencies=[Depends(require_auth)])
def model_add(req: ModelAddRequest):
    cmd = [PYTHON, str(SCRIPTS / "ft_add_model.py"), req.hf_model_id]
    if req.local_name:
        cmd += ["--name", req.local_name]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=7200)
    if result.returncode != 0:
        return err_resp("ft add-model failed", result.stderr[-1000:])
    return ok_resp({"output": result.stdout[-2000:]})


@app.get("/api/models", dependencies=[Depends(require_auth)])
def models_list():
    return ok_resp(_list_models())


@app.delete("/api/models/{name}", dependencies=[Depends(require_auth)])
def model_delete(name: str):
    _safe_name(name)
    import shutil as _sh
    p = ROOT / "models" / name
    if not p.exists():
        raise HTTPException(404, f"Model '{name}' not found")
    _sh.rmtree(str(p))
    return ok_resp({"deleted": name})


# ── Training ──────────────────────────────────────────────────────────────────

class TrainingStartRequest(BaseModel):
    model:         str   = Field(...,   max_length=200)
    dataset:       str   = Field(...,   max_length=100)
    mode:          str   = Field("vl",  pattern=r"^(vl|text)$")
    quant:         str   = Field("bf16",pattern=r"^(bf16|4bit|8bit)$")
    lora_r:        int   = Field(16,    ge=1,   le=256)
    lora_alpha:    int   = Field(32,    ge=1,   le=512)
    epochs:        int   = Field(3,     ge=1,   le=100)
    batch_size:    int   = Field(1,     ge=1,   le=32)
    learning_rate: float = Field(2e-4,  ge=1e-7, le=1.0)
    max_steps:     int   = Field(0,     ge=0,   le=1_000_000)
    wandb_project: str   = Field("",   max_length=100)
    wandb_entity:  str   = Field("",   max_length=100)
    smoke:         bool  = False


@app.post("/api/training/start", status_code=202, dependencies=[Depends(require_auth)])
def training_start(req: TrainingStartRequest):
    global _active_job

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    run_id = f"run_{ts}"
    log_path = ROOT / "logs" / f"train_{run_id}.log"

    if "/" not in req.model:
        _safe_name(req.model)
    _safe_name(req.dataset)
    model_path = str(ROOT / "models" / req.model) if "/" not in req.model else req.model
    ds_path    = str(ROOT / "datasets" / req.dataset)

    if req.mode == "vl":
        trainer_script = str(ROOT / "vl_trainer.py")
        cmd = [PYTHON, trainer_script]
        cmd += [
            "--model",       model_path,
            "--dataset",     ds_path,
            "--output",      str(ROOT / "checkpoints" / run_id),
            "--quant",       req.quant,
            "--lora-r",      str(req.lora_r),
            "--lora-alpha",  str(req.lora_alpha),
            "--epochs",      str(req.epochs),
            "--batch-size",  str(req.batch_size),
            "--lr",          str(req.learning_rate),
        ]
        if req.max_steps:
            cmd += ["--max-steps", str(req.max_steps)]
        if req.wandb_project:
            cmd += ["--wandb-project", req.wandb_project]
        if req.wandb_entity:
            cmd += ["--wandb-entity", req.wandb_entity]
        if req.smoke:
            cmd.append("--smoke")
    else:
        # Text path via Axolotl — first generate config, then train
        _safe_name(req.model)
        cfg_cmd = [PYTHON, str(SCRIPTS / "ft_configure.py"),
                   f"--model={req.model}", f"--dataset={req.dataset}",
                   f"--quantize={'4bit' if req.quant == '4bit' else 'none'}",
                   f"--epochs={req.epochs}", f"--lr={req.learning_rate}"]
        if req.wandb_project:
            cfg_cmd += [f"--wandb-project={req.wandb_project}"]
        if req.wandb_entity:
            cfg_cmd += [f"--wandb-entity={req.wandb_entity}"]
        cfg_result = subprocess.run(cfg_cmd, capture_output=True, text=True,
                                    cwd=str(ROOT), timeout=60)
        if cfg_result.returncode != 0:
            return err_resp("ft configure failed", cfg_result.stderr[-500:])

        # Find the latest config file
        cfg_dir = ROOT / "configs"
        configs = sorted(cfg_dir.glob("*.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not configs:
            configs = sorted(cfg_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not configs:
            return err_resp("No config file generated by ft configure")
        latest_cfg = str(configs[0])

        cmd = [PYTHON, str(SCRIPTS / "ft_train.py"), latest_cfg]
        if req.max_steps:
            cmd += ["--max-steps", str(req.max_steps)]

    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
           "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
           "HF_ENABLE_PARALLEL_LOADING": "0",
           "HF_HUB_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false",
           "PYTHONUNBUFFERED": "1"}

    # C2: check + Popen + assign all under one lock — eliminates TOCTOU window.
    with _active_job_lock:
        if _active_job and _active_job.is_running():
            return err_resp("A training job is already running",
                            f"run_id={_active_job.run_id}")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(ROOT), env=env,
        )
        job = _Job(run_id=run_id, proc=proc, log_path=log_path)
        job.start_monitor()
        _active_job = job

    return ok_resp({"run_id": run_id, "status": "started"})


@app.post("/api/training/stop", dependencies=[Depends(require_auth)])
def training_stop():
    with _active_job_lock:
        job = _active_job
        if job is None or not job.is_running():
            return err_resp("No training job is running")
        job.proc.terminate()
    return ok_resp({"stopped": job.run_id})


@app.get("/api/training/status", dependencies=[Depends(require_auth)])
def training_status():
    with _active_job_lock:
        job = _active_job
    if job is None:
        return ok_resp({"running": False})
    gpus = _get_gpu_info()
    vram_used = gpus[0]["vram_used_gb"] if gpus else 0.0
    return ok_resp({
        "running":          job.is_running(),
        "run_id":           job.run_id,
        "step":             job.step,
        "total_steps":      job.total_steps,
        "loss":             job.loss,
        "learning_rate":    job.lr,
        "epoch":            job.epoch,
        "elapsed_seconds":  round(job.elapsed()),
        "eta_seconds":      (round((job.elapsed() / job.step) * (job.total_steps - job.step))
                             if job.step > 0 and job.total_steps > 0 else None),
        "gpu_vram_used_gb": vram_used,
        "phase":            job.phase,
    })


@app.get("/api/training/logs", dependencies=[Depends(require_auth)])
async def training_logs(request: Request, run_id: str = "", offset: int = 0):
    """Stream log lines from offset n.  Returns text/event-stream (SSE)."""
    with _active_job_lock:
        job = _active_job

    log_path: Path | None = None
    if run_id:
        _safe_name(run_id)
        log_path = ROOT / "logs" / f"train_{run_id}.log"
    elif job:
        log_path = job.log_path

    if log_path is None or not log_path.exists():
        raise HTTPException(404, "Log file not found")

    async def _stream() -> AsyncGenerator[str, None]:
        pos = 0
        skipped = 0
        with log_path.open() as fh:
            while True:
                if await request.is_disconnected():
                    break
                line = fh.readline()
                if not line:
                    if job and job.is_running():
                        await asyncio.sleep(0.5)
                        continue
                    break
                if skipped < offset:
                    skipped += 1
                    continue
                yield f"data: {json.dumps({'line': line.rstrip(), 'pos': pos})}\n\n"
                pos += 1

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvalStartRequest(BaseModel):
    checkpoint:    str
    base_model:    str | None = None
    suite:         str = "configs/eval_default.json"
    skip_mt_bench: bool = True


@app.post("/api/eval/start", status_code=202, dependencies=[Depends(require_auth)])
def eval_start(req: EvalStartRequest):
    run_id = f"eval_{datetime.now().strftime('%Y-%m-%d_%H%M')}"
    cmd = [PYTHON, str(SCRIPTS / "ft_eval.py"), req.checkpoint,
           "--suite", req.suite]
    if req.base_model:
        cmd += ["--base", req.base_model]

    log_path = ROOT / "logs" / f"{run_id}.log"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT),
    )
    job = _BgJob(run_id=run_id, kind="eval", proc=proc, log_path=log_path)
    job.start_tail()
    with _bg_jobs_lock:
        _bg_jobs[run_id] = job
    return ok_resp({"run_id": run_id, "status": "started"})


@app.get("/api/eval/status/{run_id}", dependencies=[Depends(require_auth)])
def eval_status(run_id: str):
    _safe_name(run_id)
    with _bg_jobs_lock:
        job = _bg_jobs.get(run_id)
    if job is None:
        raise HTTPException(404, f"No eval job with run_id '{run_id}'")
    return ok_resp(job.status_dict())


@app.get("/api/eval/results/{run_id}", dependencies=[Depends(require_auth)])
def eval_results(run_id: str):
    _safe_name(run_id)
    eval_out = ROOT / "eval_results" / run_id
    if not eval_out.exists():
        raise HTTPException(404, f"Eval results for '{run_id}' not found")
    summary = eval_out / "summary.md"
    harness = eval_out / "lm_eval_harness.json"
    vqa     = eval_out / "vqa_accuracy.json"
    data: dict = {"run_id": run_id}
    if summary.exists():
        data["summary_md"] = summary.read_text()
    if harness.exists():
        data["harness"] = json.loads(harness.read_text())
    if vqa.exists():
        data["vqa"] = json.loads(vqa.read_text())
    return ok_resp(data)


# ── Export & Publish ──────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    checkpoint: str
    quantize:   str = "none"   # "awq" | "gguf" | "none"
    merge:      bool = True


@app.post("/api/export", status_code=202, dependencies=[Depends(require_auth)])
def export_start(req: ExportRequest):
    run_id = f"export_{datetime.now().strftime('%Y-%m-%d_%H%M')}"
    cmd = [PYTHON, str(SCRIPTS / "ft_export.py"), req.checkpoint]
    if req.merge:
        cmd.append("--merge")
    if req.quantize and req.quantize != "none":
        cmd += [f"--quantize={req.quantize}"]
    log_path = ROOT / "logs" / f"{run_id}.log"

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(ROOT),
    )
    job = _BgJob(run_id=run_id, kind="export", proc=proc, log_path=log_path)
    job.start_tail()
    with _bg_jobs_lock:
        _bg_jobs[run_id] = job
    return ok_resp({"run_id": run_id, "status": "started", "log": str(log_path)})


@app.get("/api/export/status/{run_id}", dependencies=[Depends(require_auth)])
def export_job_status(run_id: str):
    _safe_name(run_id)
    with _bg_jobs_lock:
        job = _bg_jobs.get(run_id)
    if job is None:
        raise HTTPException(404, f"No export job with run_id '{run_id}'")
    return ok_resp(job.status_dict())


@app.get("/api/export/status", dependencies=[Depends(require_auth)])
def export_status():
    exports = []
    d = ROOT / "exports"
    if d.exists():
        for p in sorted(d.iterdir(), reverse=True):
            merged = p / "merged"
            exports.append({
                "run":    p.name,
                "merged": merged.exists() and any(merged.glob("model*.safetensors")),
            })
    return ok_resp({"exports": exports})


class PublishRequest(BaseModel):
    export_dir:  str
    hf_repo:     str
    license:     str = "apache-2.0"
    description: str = ""


@app.post("/api/publish", dependencies=[Depends(require_auth)])
def publish(req: PublishRequest):
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise HTTPException(status_code=503, detail="HF_TOKEN env var not set on server")
    import huggingface_hub as _hfhub
    _hfhub.login(token=hf_token, add_to_git_credential=False)
    cmd = [PYTHON, str(SCRIPTS / "ft_publish.py"), req.export_dir,
           "--hf-repo", req.hf_repo,
           "--acknowledge-contamination", "--accept-base-license"]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(ROOT), timeout=3600)
    if result.returncode != 0:
        return err_resp("ft publish failed", result.stderr[-1000:])
    return ok_resp({"output": result.stdout[-2000:], "hf_repo": req.hf_repo})


# ── Inference / Chat ──────────────────────────────────────────────────────────

# In-process model state (shared with Gradio app when mounted together)
_inf_model      = None
_inf_processor  = None
_inf_model_path = ""
_inf_lock       = threading.Lock()


class ChatLoadRequest(BaseModel):
    model_path: str


@app.post("/api/chat/load", dependencies=[Depends(require_auth)])
def chat_load(req: ChatLoadRequest):
    global _inf_model, _inf_processor, _inf_model_path
    with _inf_lock:
        if _inf_model_path == req.model_path and _inf_model is not None:
            return ok_resp({"loaded": req.model_path, "already_loaded": True})
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM
        _inf_processor = AutoProcessor.from_pretrained(
            req.model_path, trust_remote_code=True
        )
        _inf_model = AutoModelForCausalLM.from_pretrained(
            req.model_path,
            device_map={"": "cuda:0"},
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        _inf_model.eval()
        _inf_model_path = req.model_path
    return ok_resp({"loaded": req.model_path})


@app.post("/api/chat/unload", dependencies=[Depends(require_auth)])
def chat_unload():
    global _inf_model, _inf_processor, _inf_model_path
    with _inf_lock:
        if _inf_model is not None:
            del _inf_model
            del _inf_processor
            _inf_model = None
            _inf_processor = None
            _inf_model_path = ""
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
    return ok_resp({"unloaded": True})


class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    model_path:   str           = Field("",  max_length=500)
    messages:     list[ChatMessage]
    image_base64: str | None    = Field(None, max_length=10_000_000)
    max_tokens:   int           = Field(256,  ge=1, le=4096)
    stream:       bool          = False


@app.post("/api/chat", dependencies=[Depends(require_auth)])
async def chat(req: ChatRequest):
    with _inf_lock:
        model     = _inf_model
        processor = _inf_processor

    if model is None:
        raise HTTPException(400, "No model loaded. Call POST /api/chat/load first.")

    import torch

    t0 = time.time()

    # Decode image if provided
    img = None
    if req.image_base64:
        from PIL import Image as _PIL
        import io
        raw = base64.b64decode(req.image_base64)
        img = _PIL.open(io.BytesIO(raw)).convert("RGB")

    msgs = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        text = processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = " ".join(m.content for m in req.messages)

    with torch.no_grad():
        if img is not None:
            inputs = processor(text=text, images=[img], return_tensors="pt").to("cuda:0")
        else:
            inputs = processor(text=text, return_tensors="pt").to("cuda:0")
        prompt_len = inputs["input_ids"].shape[1]

        if req.stream:
            from transformers import TextIteratorStreamer

            streamer = TextIteratorStreamer(
                processor.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            def _gen():
                model.generate(
                    **inputs,
                    max_new_tokens=req.max_tokens,
                    do_sample=False,
                    repetition_penalty=1.1,
                    streamer=streamer,
                )

            threading.Thread(target=_gen, daemon=True).start()

            async def _sse():
                for token in streamer:
                    yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

            return StreamingResponse(_sse(), media_type="text/event-stream")

        out_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    reply = processor.tokenizer.decode(
        out_ids[0][prompt_len:], skip_special_tokens=True
    ).strip()

    latency_ms = round((time.time() - t0) * 1000)
    tokens_gen  = out_ids.shape[1] - prompt_len

    return ok_resp({
        "role":             "assistant",
        "content":          reply,
        "tokens_generated": tokens_gen,
        "latency_ms":       latency_ms,
    })


# ═════════════════════════════════════════════════════════════════════════════
# Optional: mount Gradio on same server
# ─────────────────────────────────────────────────────────────────────────────
# If you want a single-port setup (FastAPI on 8000, Gradio at /ui):
#
#   import gradio as gr
#   import sys; sys.path.insert(0, str(ROOT))
#   import app as _gradio_app           # imports the Gradio Blocks object
#   app = gr.mount_gradio_app(app, _gradio_app.app, path="/ui")
#
# Then start with: uvicorn api:app --port 8000
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
