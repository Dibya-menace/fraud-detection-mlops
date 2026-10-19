from fraud.data import time_split


def test_split_sizes_and_fractions(raw_df):
    tr, va, te = time_split(raw_df)
    assert len(tr) + len(va) + len(te) == len(raw_df)
    assert abs(len(tr) / len(raw_df) - 0.70) < 0.01


def test_no_time_leakage(raw_df):
    tr, va, te = time_split(raw_df)
    assert tr["Time"].max() <= va["Time"].min()
    assert va["Time"].max() <= te["Time"].min()


def test_split_ignores_row_order(raw_df):
    a = time_split(raw_df)
    b = time_split(raw_df.sample(frac=1, random_state=1))
    for x, y in zip(a, b):
        assert x["Time"].tolist() == y["Time"].tolist()