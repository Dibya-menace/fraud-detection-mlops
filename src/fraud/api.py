import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODELS_DIR / "champion.joblib")
    state["meta"] = json.loads((MODELS_DIR / "metadata.json").read_text())
    yield
    state.clear()


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)


class Transaction(BaseModel):
    features: dict[str, float]


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/features")
def features():
    return {"features": state["meta"]["features"]}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    expected = state["meta"]["features"]
    missing = [f for f in expected if f not in tx.features]
    extra = [f for f in tx.features if f not in expected]
    if missing or extra:
        raise HTTPException(422, detail={"missing": missing, "unexpected": extra})

    df = pd.DataFrame([tx.features])[expected]
    model = state["model"]
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(df)[:, 1][0])
    else:
        proba = float(model.predict(df)[0])

    threshold = state["meta"]["threshold"]
    return Prediction(fraud_probability=proba,
                      is_fraud=proba >= threshold,
                      threshold=threshold)