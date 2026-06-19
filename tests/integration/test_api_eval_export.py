"""Integration tests for eval/export/training-stop API endpoints."""
from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import api as _api
    _api._API_SECRET = ""
    return TestClient(_api.app, headers={"Host": "localhost"})


def _mock_popen_empty():
    """Popen mock whose stdout yields nothing and wait() returns 0."""
    m = MagicMock()
    m.stdout = iter([])
    m.wait.return_value = 0
    return m


# ── Eval start ────────────────────────────────────────────────────────────────

def test_eval_start_success(client, tmp_path):
    import api as _api
    (tmp_path / "logs").mkdir()
    with patch.object(_api, "ROOT", tmp_path), \
         patch("subprocess.Popen", return_value=_mock_popen_empty()):
        resp = client.post("/api/eval/start", json={"checkpoint": "run_001"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["ok"] is True
    assert "run_id" in body["data"]


def test_eval_start_missing_checkpoint(client):
    resp = client.post("/api/eval/start", json={})
    assert resp.status_code == 422


def test_eval_start_with_base_model(client, tmp_path):
    import api as _api
    (tmp_path / "logs").mkdir()
    with patch.object(_api, "ROOT", tmp_path), \
         patch("subprocess.Popen", return_value=_mock_popen_empty()):
        resp = client.post("/api/eval/start", json={
            "checkpoint": "run_001",
            "base_model": "mymodel",
        })
    assert resp.status_code == 202


# ── Eval results ──────────────────────────────────────────────────────────────

def test_eval_results_not_found(client):
    resp = client.get("/api/eval/results/nonexistent-run")
    assert resp.status_code == 404


def test_eval_results_with_summary(client, tmp_path):
    import api as _api
    run = "eval_2026-01-01_1200"
    eval_dir = tmp_path / "eval_results" / run
    eval_dir.mkdir(parents=True)
    (eval_dir / "summary.md").write_text("# Eval Summary\n- accuracy: 0.82")

    with patch.object(_api, "ROOT", tmp_path):
        resp = client.get(f"/api/eval/results/{run}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "summary_md" in body["data"]


def test_eval_results_with_vqa_json(client, tmp_path):
    import api as _api
    run = "eval_2026-01-01_1300"
    eval_dir = tmp_path / "eval_results" / run
    eval_dir.mkdir(parents=True)
    (eval_dir / "vqa_accuracy.json").write_text(
        json.dumps({"before": 0.35, "after": 0.60, "n": 50})
    )

    with patch.object(_api, "ROOT", tmp_path):
        resp = client.get(f"/api/eval/results/{run}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["data"]["vqa"]["before"] == pytest.approx(0.35)


# ── Export start ──────────────────────────────────────────────────────────────

def test_export_start_success(client, tmp_path):
    import api as _api
    (tmp_path / "logs").mkdir()
    with patch.object(_api, "ROOT", tmp_path), \
         patch("subprocess.Popen", return_value=_mock_popen_empty()):
        resp = client.post("/api/export", json={"checkpoint": "run_001"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["ok"] is True
    assert "run_id" in body["data"]


def test_export_start_with_quantize(client, tmp_path):
    import api as _api
    (tmp_path / "logs").mkdir()
    with patch.object(_api, "ROOT", tmp_path), \
         patch("subprocess.Popen", return_value=_mock_popen_empty()):
        resp = client.post("/api/export", json={
            "checkpoint": "run_001",
            "quantize": "awq",
            "merge": True,
        })
    assert resp.status_code == 202


# ── Export status ─────────────────────────────────────────────────────────────

def test_export_status_empty(client, tmp_path):
    import api as _api
    (tmp_path / "exports").mkdir()
    with patch.object(_api, "ROOT", tmp_path):
        resp = client.get("/api/export/status")
    assert resp.status_code == 200
    assert resp.json()["data"]["exports"] == []


def test_export_status_with_merged(client, tmp_path):
    import api as _api
    run_dir = tmp_path / "exports" / "export_2026-01-01_1200"
    merged  = run_dir / "merged"
    merged.mkdir(parents=True)
    (merged / "model.safetensors").write_bytes(b"fake")

    with patch.object(_api, "ROOT", tmp_path):
        resp = client.get("/api/export/status")
    assert resp.status_code == 200
    exports = resp.json()["data"]["exports"]
    assert len(exports) == 1
    assert exports[0]["merged"] is True


# ── Training stop ─────────────────────────────────────────────────────────────

def test_training_stop_no_job(client):
    import api as _api
    prev = _api._active_job
    _api._active_job = None
    try:
        resp = client.post("/api/training/stop")
        # err_resp returns 400 when no job is running
        assert resp.status_code == 400
        assert resp.json()["ok"] is False
    finally:
        _api._active_job = prev


def test_training_stop_with_running_job(client):
    import api as _api
    prev = _api._active_job
    mock_job = MagicMock()
    mock_job.is_running.return_value = True
    mock_job.run_id = "run_test"
    _api._active_job = mock_job
    try:
        resp = client.post("/api/training/stop")
        assert resp.status_code == 200
    finally:
        _api._active_job = prev


# ── Training logs streaming ───────────────────────────────────────────────────

def test_training_logs_no_log_file(client, tmp_path):
    import api as _api
    (tmp_path / "logs").mkdir()
    with patch.object(_api, "ROOT", tmp_path):
        resp = client.get("/api/training/logs?run_id=nonexistent")
    # Raises 404 when log file doesn't exist
    assert resp.status_code == 404


def test_training_logs_with_file(client, tmp_path):
    import api as _api
    (tmp_path / "logs").mkdir()
    log = tmp_path / "logs" / "train_myrun.log"
    log.write_text("step 1\nstep 2\n")
    with patch.object(_api, "ROOT", tmp_path):
        resp = client.get("/api/training/logs?run_id=myrun")
    assert resp.status_code == 200


# ── Chat unload ───────────────────────────────────────────────────────────────

def test_chat_unload_no_model(client):
    resp = client.post("/api/chat/unload")
    assert resp.status_code == 200
