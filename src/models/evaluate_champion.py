"""Evaluate the registered champion on the untouched test set and log it to its MLflow run."""
import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

cfg = yaml.safe_load(open("configs/config.yaml"))
d, m = cfg["data"], cfg["model"]
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

name, alias, thr = m["registered_name"], m["champion_alias"], m["threshold"]
mv = MlflowClient().get_model_version_by_alias(name, alias)
model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")

test = pd.read_parquet(d["test"])
features = [c for c in test.columns if c not in d["drop_cols"]]
y = test[d["target"]].values
amount = test["Amount"].values
p = model.predict_proba(test[features])[:, 1]
flagged = p >= thr

metrics = {
    "test_pr_auc": average_precision_score(y, p),
    "test_roc_auc": roc_auc_score(y, p),
    "test_precision_at_thr": precision_score(y, flagged, zero_division=0),
    "test_recall_at_thr": recall_score(y, flagged),
    "test_flagged": int(flagged.sum()),
    "test_cost_at_thr": float(amount[(y == 1) & ~flagged].sum() + m["review_cost"] * ((y == 0) & flagged).sum()),
    "test_cost_no_model": float(amount[y == 1].sum()),
}
with mlflow.start_run(run_id=mv.run_id):
    mlflow.log_metrics(metrics)

print(f"Champion: {name} v{mv.version} (run {mv.run_id[:8]}), threshold {thr}")
for k, v in metrics.items():
    print(f"  {k:24s} {v:,.4f}")