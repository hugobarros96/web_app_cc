from fastapi.testclient import TestClient

from portfolio.app import app

client = TestClient(app)


def test_datadoctor_route_serves_iframe(monkeypatch):
    monkeypatch.setenv("DATADOCTOR_URL", "https://example-datadoctor.hf.space")
    resp = client.get("/datadoctor")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<iframe" in resp.text
    assert "https://example-datadoctor.hf.space" in resp.text
