"""
Smoke test for the Week 5 triage service.

Uses FastAPI's TestClient (no running server needed) and runs fully offline
against the LocalFallbackChat backend -- proving the endpoint works end to end.

Run:
    pytest -v
"""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_triage_billing_message():
    resp = client.post(
        "/triage",
        json={"message": "I was charged twice this month, please refund me"},
    )
    assert resp.status_code == 200
    body = resp.json()
    # Structure.
    assert set(body) == {
        "category", "priority", "suggested_reply", "retrieved", "backend"
    }
    # Routing: a double-charge/refund message is billing + high priority.
    assert body["category"] == "billing"
    assert body["priority"] == "high"
    # The reply is grounded: at least one KB doc was retrieved.
    assert isinstance(body["retrieved"], list)
    assert len(body["retrieved"]) >= 1
    assert body["suggested_reply"]


def test_triage_technical_message():
    resp = client.post(
        "/triage",
        json={"message": "The app crashes with a 500 error when I log in"},
    )
    assert resp.status_code == 200
    assert resp.json()["category"] == "technical"


def test_triage_validation_rejects_empty():
    resp = client.post("/triage", json={"message": ""})
    assert resp.status_code == 422  # pydantic min_length violation
