"""Security tests — API auth middleware."""
from __future__ import annotations

import os
import pytest


@pytest.fixture()
def client_no_auth(monkeypatch):
    monkeypatch.delenv("API_SECRET", raising=False)
    import api as _api
    monkeypatch.setattr(_api, "_API_SECRET", "")
    from fastapi.testclient import TestClient
    return TestClient(_api.app, headers={"Host": "localhost"})


@pytest.fixture()
def client_with_auth(monkeypatch):
    secret = "super-secret-test-token"
    import api as _api
    monkeypatch.setattr(_api, "_API_SECRET", secret)
    from fastapi.testclient import TestClient
    return TestClient(_api.app, headers={"Host": "localhost"}), secret


def test_health_always_public(client_no_auth):
    resp = client_no_auth.get("/api/health")
    assert resp.status_code == 200


def test_status_always_public(client_no_auth):
    resp = client_no_auth.get("/api/status")
    assert resp.status_code == 200


def test_no_auth_secret_allows_all(client_no_auth):
    """When API_SECRET is not set, every endpoint is accessible."""
    resp = client_no_auth.get("/api/datasets")
    assert resp.status_code == 200


def test_missing_bearer_token_rejected(client_with_auth):
    client, _ = client_with_auth
    resp = client.get("/api/datasets")
    assert resp.status_code == 401


def test_wrong_bearer_token_rejected(client_with_auth):
    client, _ = client_with_auth
    resp = client.get("/api/datasets", headers={"Authorization": "Bearer wrong-token"})
    assert resp.status_code == 403


def test_correct_bearer_token_accepted(client_with_auth):
    client, secret = client_with_auth
    resp = client.get("/api/datasets", headers={"Authorization": f"Bearer {secret}"})
    assert resp.status_code == 200


def test_malformed_auth_header_rejected(client_with_auth):
    client, _ = client_with_auth
    resp = client.get("/api/datasets", headers={"Authorization": "Basic dXNlcjpwYXNz"})
    assert resp.status_code == 401


def test_empty_auth_header_rejected(client_with_auth):
    client, _ = client_with_auth
    resp = client.get("/api/datasets", headers={"Authorization": ""})
    assert resp.status_code == 401
