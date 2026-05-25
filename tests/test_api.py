from fastapi.testclient import TestClient
import os
from app.main import app
from tests.test_classifier import DummyClassifier

os.environ["TESTING"] = "true"
client = TestClient(app)
app.state.classifier = DummyClassifier()

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"]


def test_metrics():
    response = client.get("/metrics")

    assert response.status_code == 200
    data = response.json()

    assert "total_requests" in data
    assert "successful_predictions" in data
    assert "invalid_images" in data
    assert "failures" in data
    assert "total_latency" in data


def test_predict():
    with open("tests/cucumber.jpg", "rb") as f:
        response = client.post(
            "/predict",
            files={"file": ("cucumber.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data
    assert "latency" in data
    assert data["prediction"] == "Cucumber"
    assert data["confidence"] == 0.99
