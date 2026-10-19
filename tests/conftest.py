import numpy as np
import pandas as pd
import pytest

from fraud.config import V_COLS


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLS)
    df.insert(0, "Time", np.arange(n, dtype=float) * 10)
    df["Amount"] = rng.uniform(1, 200, n)
    df["Class"] = (rng.random(n) < 0.1).astype(int)
    df.loc[df["Class"] == 1, "V1"] += 3   # make fraud learnable
    return df