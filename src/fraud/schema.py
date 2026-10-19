from pandera.pandas import Check, Column, DataFrameSchema

from fraud.config import V_COLS

_feature_cols = {
    "Time": Column(float, Check.ge(0), nullable=False),
    **{c: Column(float, nullable=False) for c in V_COLS},
    "Amount": Column(float, Check.ge(0), nullable=False),
}

# Labelled data (training / evaluation)
RAW_SCHEMA = DataFrameSchema(
    {**_feature_cols, "Class": Column(int, Check.isin([0, 1]), nullable=False)},
    strict=True,   # unexpected extra columns are an error
    coerce=True,
)

# Unlabelled data (what the API will receive in Step 6)
INFERENCE_SCHEMA = DataFrameSchema(_feature_cols, strict=True, coerce=True)


def validate_raw(df):
    return RAW_SCHEMA.validate(df)


def validate_inference(df):
    return INFERENCE_SCHEMA.validate(df)