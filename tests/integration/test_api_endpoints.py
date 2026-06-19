"""Integration tests — FastAPI endpoint behaviour via TestClient."""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import api as _api
    _api._API_SECRET = ""   # disable auth for integration tests
    return TestClient(_api.app, headers={"Host": "localhost"})


# ── Health / Status ───────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True


def test_status(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert "active_job" in body["data"]


# ── Datasets ──────────────────────────────────────────────────────────────────

def test_datasets_list(client):
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_datasets_add_missing_source(client):
    resp = client.post("/api/datasets/add", json={})
    assert resp.status_code == 422   # Pydantic validation error


def test_datasets_add_source_too_long(client):
    resp = client.post("/api/datasets/add", json={"source": "x" * 501})
    assert resp.status_code == 422


def test_datasets_clean_bad_val_ratio(client):
    resp = client.post("/api/datasets/clean", json={
        "name": "good-name", "val_ratio": 1.5
    })
    assert resp.status_code == 422


def test_datasets_clean_negative_max_chars(client):
    resp = client.post("/api/datasets/clean", json={
        "name": "good-name", "max_chars": -1
    })
    assert resp.status_code == 422


def test_datasets_sample_not_found(client):
    resp = client.get("/api/datasets/no-such-dataset/sample")
    assert resp.status_code == 404


def test_datasets_card_not_found(client):
    resp = client.get("/api/datasets/no-such-dataset/card")
    assert resp.status_code == 404


# ── Models ────────────────────────────────────────────────────────────────────

def test_models_list(client):
    resp = client.get("/api/models")
    assert resp.status_code == 200


def test_models_add_validation(client):
    resp = client.post("/api/models/add", json={"hf_model_id": "x" * 201})
    assert resp.status_code == 422


def test_model_delete_not_found(client):
    resp = client.delete("/api/models/nonexistent-model")
    assert resp.status_code == 404


# ── Training ──────────────────────────────────────────────────────────────────

def test_training_start_invalid_mode(client):
    resp = client.post("/api/training/start", json={
        "model":   "mymodel",
        "dataset": "myds",
        "mode":    "invalid_mode",
    })
    assert resp.status_code == 422


def test_training_start_invalid_quant(client):
    resp = client.post("/api/training/start", json={
        "model":   "mymodel",
        "dataset": "myds",
        "quant":   "fp16",   # not in allowed set
    })
    assert resp.status_code == 422


def test_training_start_lora_r_too_large(client):
    resp = client.post("/api/training/start", json={
        "model": "m", "dataset": "d", "lora_r": 999
    })
    assert resp.status_code == 422


def test_training_start_negative_epochs(client):
    resp = client.post("/api/training/start", json={
        "model": "m", "dataset": "d", "epochs": 0
    })
    assert resp.status_code == 422


# ── Chat ──────────────────────────────────────────────────────────────────────

def test_chat_image_too_large(client):
    resp = client.post("/api/chat", json={
        "messages":     [{"role": "user", "content": "hi"}],
        "image_base64": "x" * 10_000_001,   # over 10 MB limit
    })
    assert resp.status_code == 422


def test_chat_max_tokens_too_large(client):
    resp = client.post("/api/chat", json={
        "messages":   [{"role": "user", "content": "hi"}],
        "max_tokens": 9999,
    })
    assert resp.status_code == 422


def test_chat_max_tokens_zero(client):
    resp = client.post("/api/chat", json={
        "messages":   [{"role": "user", "content": "hi"}],
        "max_tokens": 0,
    })
    assert resp.status_code == 422


def test_chat_no_model_loaded(client):
    """Without a loaded model, /api/chat should return 400 (not 500)."""
    resp = client.post("/api/chat", json={
        "messages": [{"role": "user", "content": "hello"}]
    })
    assert resp.status_code == 400


# ── Publish ───────────────────────────────────────────────────────────────────

def test_publish_no_hf_token_field(client):
    """PublishRequest must not accept hf_token in the body (S4 fix)."""
    import api as _api
    from pydantic import BaseModel
    # Check the actual model definition has no hf_token field
    fields = _api.PublishRequest.model_fields
    assert "hf_token" not in fields, "hf_token must not be a field on PublishRequest"
