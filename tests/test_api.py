import pytest
from fastapi.testclient import TestClient

from fraud.api import app


@pytest.fixture(scope="module")
def client():
    # "with" runs startup events, so the model gets loaded
    with TestClient(app) as c:
        yield c


def valid_features():
    feats = {f"V{i}": 0.0 for i in range(1, 29)}
    feats["Amount"] = 50.0
    return feats


def valid_payload():
    return {"features": valid_features()}


def test_predict_returns_200(client):
    r = client.post("/predict", json=valid_payload())
    assert r.status_code == 200, r.text
    assert isinstance(r.json(), dict)


def test_predict_probability_in_range(client):
    r = client.post("/predict", json=valid_payload())
    assert r.status_code == 200, r.text
    body = r.json()
    probs = [v for v in body.values() if isinstance(v, float)]
    assert probs, f"no float probability found in response: {body}"
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_missing_feature_is_rejected(client):
    feats = valid_features()
    feats.pop("V1")
    r = client.post("/predict", json={"features": feats})
    assert r.status_code in (400, 422), r.text


def test_wrong_type_is_rejected(client):
    feats = valid_features()
    feats["Amount"] = "not-a-number"
    r = client.post("/predict", json={"features": feats})
    assert r.status_code in (400, 422), r.text