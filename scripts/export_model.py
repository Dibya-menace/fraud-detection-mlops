"""Export the MLflow champion model to models/ for serving (run from project root)."""
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
MODEL_URI = "models:/fraud-lgbm@champion"
THRESHOLD = 0.50

if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def load(uri):
    try:
        return mlflow.sklearn.load_model(uri)
    except Exception as e1:
        try:
            return mlflow.lightgbm.load_model(uri)
        except Exception as e2:
            raise SystemExit(f"Could not load {uri}\nsklearn: {e1}\nlightgbm: {e2}")


def feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "booster_"):
        return list(model.booster_.feature_name())
    if hasattr(model, "feature_name"):
        return list(model.feature_name())
    raise SystemExit("Cannot read feature names from model")


model = load(MODEL_URI)
features = feature_names(model)

out = ROOT / "models"
out.mkdir(exist_ok=True)
joblib.dump(model, out / "champion.joblib")
(out / "metadata.json").write_text(
    json.dumps({"model_name": "fraud-lgbm", "alias": "champion",
                "threshold": THRESHOLD, "features": features}, indent=2)
)
print("Model type:", type(model))
print("Features:", len(features), features)
print("Saved to", out)