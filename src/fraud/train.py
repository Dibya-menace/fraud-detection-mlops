import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from fraud.config import FEATURES, RANDOM_STATE, REVIEW_COST, TARGET, THRESHOLD
from fraud.data import load_split

# REPLACE with the lgbm_sqrt params from your notebook / MLflow run
PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)


def fit(train_df, **overrides):
    y = train_df[TARGET]
    spw = float(np.sqrt((y == 0).sum() / (y == 1).sum()))
    model = LGBMClassifier(**{**PARAMS, "scale_pos_weight": spw, **overrides})
    model.fit(train_df[FEATURES], y)
    return model


def evaluate(model, df, threshold=THRESHOLD):
    proba = model.predict_proba(df[FEATURES])[:, 1]
    pred = proba >= threshold
    y = df[TARGET].to_numpy()
    missed_amount = df.loc[(~pred) & (y == 1), "Amount"].sum()
    cost = missed_amount + REVIEW_COST * pred.sum()
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "cost": float(cost),
    }


if __name__ == "__main__":
    model = fit(load_split("train"))
    for name in ("val", "test"):
        print(name, {k: round(v, 4) for k, v in evaluate(model, load_split(name)).items()})