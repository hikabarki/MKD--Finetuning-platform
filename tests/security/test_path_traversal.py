"""Security tests — path traversal via _safe_name."""
from __future__ import annotations

import pytest
from fastapi import HTTPException


def safe(s):
    from api import _safe_name
    return _safe_name(s)


# ── Rejection cases ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    "../etc/passwd",
    "../../root/.ssh/id_rsa",
    "foo/bar",
    "foo/../bar",
    "..",
    "",
    "\x00nullbyte",
    "name\x00injected",
    "/absolute",
    "a/b",
])
def test_safe_name_rejects_bad_input(bad):
    with pytest.raises(HTTPException) as exc_info:
        safe(bad)
    assert exc_info.value.status_code == 400


# ── Acceptance cases ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("good", [
    "mymodel",
    "dataset-v2",
    "run_2026-06-18_0900",
    "google__gemma-4-12b",
    "abc123",
    "my.model.ckpt",
    "A" * 50,
])
def test_safe_name_accepts_valid(good):
    result = safe(good)
    assert result == good


# ── Endpoint-level path traversal ────────────────────────────────────────────

def test_dataset_delete_traversal():
    """DELETE /api/datasets/../../../etc/passwd must be rejected (422 from FastAPI URL parsing)."""
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app, raise_server_exceptions=False, headers={"Host": "localhost"})
    # FastAPI normalises the URL before it reaches the handler; a raw traversal
    # attempt returns either 404 (normalised away) or 400 (_safe_name raised).
    resp = client.delete("/api/datasets/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404, 422)


def test_model_delete_traversal():
    from fastapi.testclient import TestClient
    from api import app
    client = TestClient(app, raise_server_exceptions=False, headers={"Host": "localhost"})
    resp = client.delete("/api/models/..%2Fetc")
    assert resp.status_code in (400, 404, 422)
