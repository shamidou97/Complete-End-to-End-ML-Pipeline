from fastapi.testclient import TestClient
from src.api import app
import os

# Create a test client (like a fake browser)
client = TestClient(app)

def test_read_root():
    """Check if the root endpoint returns a 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_health_check():
    """Check if the /health endpoint is working."""
    response = client.get("/health")
    assert response.status_code == 200
    # Note: It might return 'error' if model isn't loaded, but status code should still be 200 (OK)
    assert "status" in response.json()

def test_prediction_endpoint_exists():
    """Ensure the POST /predict endpoint is reachable."""
    # Sending a request without a file should fail with 422 (Validation Error)
    # This proves the endpoint exists and is checking inputs.
    response = client.post("/predict")
    assert response.status_code == 422
