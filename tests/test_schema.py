import pytest
from pandera.errors import SchemaError, SchemaErrors

from fraud.schema import validate_inference, validate_raw

ERRORS = (SchemaError, SchemaErrors)


def test_valid_frame_passes(raw_df):
    assert len(validate_raw(raw_df)) == len(raw_df)


def test_negative_amount_fails(raw_df):
    raw_df.loc[0, "Amount"] = -1.0
    with pytest.raises(ERRORS):
        validate_raw(raw_df)


def test_bad_label_fails(raw_df):
    raw_df.loc[0, "Class"] = 2
    with pytest.raises(ERRORS):
        validate_raw(raw_df)


def test_null_feature_fails(raw_df):
    raw_df.loc[0, "V5"] = None
    with pytest.raises(ERRORS):
        validate_raw(raw_df)


def test_missing_column_fails(raw_df):
    with pytest.raises(ERRORS):
        validate_raw(raw_df.drop(columns=["V3"]))


def test_inference_schema_has_no_label(raw_df):
    assert "Class" not in validate_inference(raw_df.drop(columns=["Class"])).columns