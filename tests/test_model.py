from fraud.train import evaluate, fit


def test_model_outputs_valid_probabilities(raw_df):
    model = fit(raw_df, n_estimators=20)
    proba = model.predict_proba(raw_df[model.feature_name_])[:, 1]
    assert proba.shape == (len(raw_df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_evaluate_returns_metrics_and_beats_base_rate(raw_df):
    model = fit(raw_df, n_estimators=20)
    m = evaluate(model, raw_df)
    assert set(m) == {"pr_auc", "roc_auc", "precision", "recall", "cost"}
    assert m["pr_auc"] > raw_df["Class"].mean()