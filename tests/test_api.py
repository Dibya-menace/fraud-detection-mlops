import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud.api import app

META = Path(__file__).resolve().parents[1] / "models" / "metadata.json"
pytestmark = pytest.mark.skipif(not META.exists(), reason="run scripts/export_model.py first")


def test_health_and_predict():
    features = json.loads(META.read_text())["features"]
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is True
        r = c.post("/predict", json={"features": {f: 0.0 for f in features}})
        assert r.status_code == 200
        assert 0.0 <= r.json()["fraud_probability"] <= 1.0


def test_rejects_bad_input():
    with TestClient(app) as c:
        r = c.post("/predict", json={"features": {"nope": 1.0}})
        assert r.status_code == 422