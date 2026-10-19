from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "creditcard.csv"   # adjust if yours differs
PROCESSED_DIR = ROOT / "data" / "processed"

TARGET = "Class"
V_COLS = [f"V{i}" for i in range(1, 29)]
# Must match the feature list in your notebook. Time is excluded here because
# raw elapsed time can't generalise under a time-based split.
FEATURES = V_COLS + ["Amount"]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15          # test = remaining 0.15
THRESHOLD = 0.50
REVIEW_COST = 5.0        # assumption: cost of manually reviewing a flagged txn
RANDOM_STATE = 42