"""Train candidates, log every run to MLflow, register the best by validation PR-AUC."""
import subprocess
import warnings

import lightgbm as lgb
import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=".*eval_set.*")  # harmless LightGBM 4.7 deprecation

cfg = yaml.safe_load(open("configs/config.yaml"))
d, m = cfg["data"], cfg["model"]
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment"])

train = pd.read_parquet(d["train"])
val = pd.read_parquet(d["val"])
features = [c for c in train.columns if c not in d["drop_cols"]]
X_tr, y_tr = train[features], train[d["target"]]
X_va, y_va = val[features], val[d["target"]]
neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def log_run(name, model, params):
    p = model.predict_proba(X_va)[:, 1]
    flagged = p >= m["threshold"]
    metrics = {
        "val_pr_auc": average_precision_score(y_va, p),
        "val_roc_auc": roc_auc_score(y_va, p),
        "val_precision_at_thr": precision_score(y_va, flagged, zero_division=0),
        "val_recall_at_thr": recall_score(y_va, flagged),
    }
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"git_sha": git_sha(), "split": "time_70_15_15", "threshold": str(m["threshold"])})
        mlflow.sklearn.log_model(model, name="model", input_example=X_va.head(5),serialization_format="cloudpickle")
    print(f"{name:12s} PR-AUC={metrics['val_pr_auc']:.4f}  ROC-AUC={metrics['val_roc_auc']:.4f}")
    return run.info.run_id, metrics["val_pr_auc"]


runs = {}

logreg = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
]).fit(X_tr, y_tr)
runs["logreg"] = log_run("logreg", logreg, {"model": "logreg", "class_weight": "balanced", "max_iter": 1000})

for tag, spw in [("lgbm_spw1", 1.0), ("lgbm_sqrt", (neg / pos) ** 0.5), ("lgbm_full", neg / pos)]:
    params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  scale_pos_weight=round(spw, 2), random_state=42)
    clf = lgb.LGBMClassifier(**params, metric="average_precision", n_jobs=-1, verbose=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100, verbose=False)])
    runs[tag] = log_run(tag, clf, {**params, "model": "lightgbm", "best_iteration": clf.best_iteration_})

best = max(runs, key=lambda k: runs[k][1])
run_id = runs[best][0]
mv = mlflow.register_model(f"runs:/{run_id}/model", m["registered_name"])
MlflowClient().set_registered_model_alias(m["registered_name"], m["champion_alias"], mv.version)
print(f"\nBest: {best} -> registered {m['registered_name']} v{mv.version} with alias '{m['champion_alias']}'")