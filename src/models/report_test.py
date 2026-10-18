"""Report-only: logreg vs champion on test, with bootstrap CIs. Does not change the champion."""
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

cfg = yaml.safe_load(open("configs/config.yaml"))
d, m = cfg["data"], cfg["model"]
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

train = pd.read_parquet(d["train"])
test = pd.read_parquet(d["test"])
features = [c for c in train.columns if c not in d["drop_cols"]]
y = test[d["target"]].values

logreg = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
]).fit(train[features], train[d["target"]])
champ = mlflow.sklearn.load_model(f"models:/{m['registered_name']}@{m['champion_alias']}")

p_lr = logreg.predict_proba(test[features])[:, 1]
p_gb = champ.predict_proba(test[features])[:, 1]
print(f"Test PR-AUC  logreg={average_precision_score(y, p_lr):.4f}  lightgbm={average_precision_score(y, p_gb):.4f}")

rng = np.random.default_rng(0)
gb_scores, diffs = [], []
for _ in range(1000):
    idx = rng.integers(0, len(y), len(y))
    if y[idx].sum() == 0:
        continue
    a = average_precision_score(y[idx], p_gb[idx])
    b = average_precision_score(y[idx], p_lr[idx])
    gb_scores.append(a)
    diffs.append(a - b)

lo, hi = np.percentile(gb_scores, [2.5, 97.5])
dlo, dhi = np.percentile(diffs, [2.5, 97.5])
print(f"LightGBM test PR-AUC 95% CI : [{lo:.3f}, {hi:.3f}]")
print(f"LightGBM minus logreg 95% CI: [{dlo:+.3f}, {dhi:+.3f}]")