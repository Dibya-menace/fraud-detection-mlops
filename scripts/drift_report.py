"""Data drift check with Evidently: training period vs later period,
plus a simulated shift to prove the check can detect drift."""
import pandas as pd
from scipy.stats import ks_2samp

from fraud.config import RAW_PATH, ROOT

OUT = ROOT / "reports" / "drift"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
N = 20_000

# Same time-based split as the model: sort by Time, first 70% = train, last 15% = test
df = pd.read_csv(RAW_PATH).sort_values("Time").reset_index(drop=True)
n = len(df)
train = df.iloc[: int(n * 0.70)]
test = df.iloc[int(n * 0.85):]

reference = train[FEATURES].sample(N, random_state=42).reset_index(drop=True)
current = test[FEATURES].sample(min(N, len(test)), random_state=42).reset_index(drop=True)

# Simulated drift: spending jumps 3x and V14 moves by 1.0
simulated = current.copy()
simulated["Amount"] = simulated["Amount"] * 3
simulated["V14"] = simulated["V14"] + 1.0


def run_evidently(ref, cur, path):
    try:  # newer Evidently
        from evidently import Report
        from evidently.presets import DataDriftPreset

        result = Report([DataDriftPreset()]).run(cur, ref)
        result.save_html(str(path))
    except ImportError:  # older Evidently
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref, current_data=cur)
        report.save_html(str(path))
    print("Saved", path)


def ks_summary(ref, cur, label):
    # KS statistic = size of the distribution gap (0 = identical, 1 = completely different).
    # With 20k rows even tiny gaps get tiny p-values, so judge by the statistic size.
    stats = {c: ks_2samp(ref[c], cur[c]).statistic for c in FEATURES}
    top = sorted(stats.items(), key=lambda kv: kv[1], reverse=True)[:5]
    material = [c for c, s in stats.items() if s > 0.10]
    print(f"\n[{label}] features with KS statistic > 0.10: {len(material)} of {len(FEATURES)}")
    print("  top 5:", ", ".join(f"{c}={s:.3f}" for c, s in top))


run_evidently(reference, current, OUT / "train_vs_test.html")
ks_summary(reference, current, "train vs test")

run_evidently(reference, simulated, OUT / "simulated_drift.html")
ks_summary(reference, simulated, "train vs simulated shift")