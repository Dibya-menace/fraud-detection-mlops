import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from fraud.config import FEATURES, RANDOM_STATE, REVIEW_COST, TARGET, THRESHOLD
from fraud.data import load_split

PARAMS = dict(
    n_estimators=2000,          # ceiling only; early stopping picks the real count
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    metric="average_precision",  # what early stopping watches
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
EARLY_STOPPING_ROUNDS = 100


def fit(train_df, val_df=None, **overrides):
    y = train_df[TARGET]
    spw = float(np.sqrt((y == 0).sum() / (y == 1).sum()))
    model = LGBMClassifier(**{**PARAMS, "scale_pos_weight": spw, **overrides})
    if val_df is None:
        model.fit(train_df[FEATURES], y)
    else:
        model.fit(
            train_df[FEATURES],
            y,
            eval_X=val_df[FEATURES],
            eval_y=val_df[TARGET],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
    return model


def evaluate(model, df, threshold=THRESHOLD):
    proba = model.predict_proba(df[FEATURES])[:, 1]
    pred = proba >= threshold
    y = df[TARGET].to_numpy()
    missed_amount = df.loc[(~pred) & (y == 1), "Amount"].sum()
    false_positives = int((pred & (y == 0)).sum())
    cost = missed_amount + REVIEW_COST * false_positives
    return {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "cost": float(cost),
    }


if __name__ == "__main__":
    train, val = load_split("train"), load_split("val")
    model = fit(train, val)
    print("best_iteration:", model.best_iteration_)
    for name, df in (("val", val), ("test", load_split("test"))):
        print(name, {k: round(v, 4) for k, v in evaluate(model, df).items()})