import pandas as pd

from fraud.config import PROCESSED_DIR, RAW_PATH, TRAIN_FRAC, VAL_FRAC
from fraud.schema import validate_raw


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    return validate_raw(pd.read_csv(path))


def time_split(df: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Chronological split. Sorting first means row order can't cause leakage."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    i, j = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i], df.iloc[i:j], df.iloc[j:]


def load_split(name: str) -> pd.DataFrame:
    """name in {'train', 'val', 'test'}; re-validated on load."""
    return validate_raw(pd.read_parquet(PROCESSED_DIR / f"{name}.parquet"))