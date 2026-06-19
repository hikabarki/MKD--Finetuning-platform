"""Integration tests for API endpoints that call subprocess — fully mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import api as _api
    _api._API_SECRET = ""
    return TestClient(_api.app, headers={"Host": "localhost"})


def _ok_proc():
    """Fake CompletedProcess with rc=0."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = "done"
    m.stderr = ""
    return m


def _fail_proc():
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    m.stderr = "error occurred"
    return m


# ── Dataset endpoints ─────────────────────────────────────────────────────────

def test_dataset_add_success(client):
    with patch("subprocess.run", return_value=_ok_proc()):
        resp = client.post("/api/datasets/add", json={"source": "hf://some/dataset"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_dataset_add_failure(client):
    with patch("subprocess.run", return_value=_fail_proc()):
        resp = client.post("/api/datasets/add", json={"source": "hf://some/dataset"})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


def test_dataset_clean_success(client):
    with patch("subprocess.run", return_value=_ok_proc()):
        resp = client.post("/api/datasets/clean", json={"name": "my-dataset"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_dataset_clean_invalid_name_slash(client):
    resp = client.post("/api/datasets/clean", json={"name": "foo/bar"})
    assert resp.status_code == 400


def test_dataset_delete_success(client, tmp_path):
    import api as _api
    ds_dir = tmp_path / "datasets" / "to-delete"
    ds_dir.mkdir(parents=True)
    with patch.object(_api, "ROOT", tmp_path):
        resp = client.delete("/api/datasets/to-delete")
    assert resp.status_code == 200


# ── Model endpoints ───────────────────────────────────────────────────────────

def test_model_add_success(client):
    with patch("subprocess.run", return_value=_ok_proc()):
        resp = client.post("/api/models/add", json={"hf_model_id": "nvidia/some-model"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_model_add_failure(client):
    with patch("subprocess.run", return_value=_fail_proc()):
        resp = client.post("/api/models/add", json={"hf_model_id": "nvidia/some-model"})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False


# ── Training endpoint ─────────────────────────────────────────────────────────

def test_training_start_success(client, tmp_path):
    import api as _api
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "datasets" / "myds").mkdir(parents=True)
    (tmp_path / "models" / "mymodel").mkdir(parents=True)

    mock_proc = MagicMock()
    mock_proc.pid = 12345

    with patch.object(_api, "ROOT", tmp_path), \
         patch("subprocess.Popen", return_value=mock_proc):
        resp = client.post("/api/training/start", json={
            "model":   "mymodel",
            "dataset": "myds",
        })
    assert resp.status_code == 202
    assert resp.json()["ok"] is True


def test_training_start_already_running(client, tmp_path):
    import api as _api
    prev_job = _api._active_job
    mock_job = MagicMock()
    mock_job.is_running.return_value = True
    mock_job.run_id = "existing"
    _api._active_job = mock_job
    try:
        resp = client.post("/api/training/start", json={
            "model": "m", "dataset": "d"
        })
        assert resp.status_code == 400
    finally:
        _api._active_job = prev_job


def test_training_status(client):
    resp = client.get("/api/training/status")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ── Export / Eval ─────────────────────────────────────────────────────────────

def test_eval_results_not_found(client):
    resp = client.get("/api/eval/results/no-such-run")
    assert resp.status_code in (200, 404)


def test_export_list(client):
    resp = client.get("/api/export/status")
    assert resp.status_code == 200


# ── Publish (no hf_token) ─────────────────────────────────────────────────────

def test_publish_missing_hf_token_env(client, monkeypatch):
    """Without HF_TOKEN, publish returns 503 (service unavailable)."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    resp = client.post("/api/publish", json={
        "export_dir": "exports/run_test/merged",
        "hf_repo":    "mkd-hika/test-model",
    })
    assert resp.status_code == 503


def test_publish_with_hf_token_subprocess_failure(client, monkeypatch):
    """With HF_TOKEN set, subprocess failure → 400."""
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token")
    with patch("subprocess.run", return_value=_fail_proc()), \
         patch("huggingface_hub.login"):   # skip real HF validation
        resp = client.post("/api/publish", json={
            "export_dir": "exports/run_test/merged",
            "hf_repo":    "mkd-hika/test-model",
        })
    assert resp.status_code == 400
    assert resp.json()["ok"] is False
