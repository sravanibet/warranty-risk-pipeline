import pandas as pd

from warranty_risk.agent.data_validator import validate_upload


def test_valid_upload_passes(valid_config):
    df = pd.DataFrame({
        "SNS_A1": [1.0, 2.0, 3.0],
        "SNS_B1": [0, 1, 0],
        "claim": [0, 1, 0],
    })
    result = validate_upload(df, valid_config, label_col="claim")
    assert result.is_valid
    assert result.missing_sensors == []


def test_missing_sensor_detected(valid_config):
    df = pd.DataFrame({
        "SNS_A1": [1.0, 2.0, 3.0],
        "claim": [0, 1, 0],
    })
    result = validate_upload(df, valid_config, label_col="claim")
    assert not result.is_valid
    assert "SNS_B1" in result.missing_sensors


def test_extra_column_detected(valid_config):
    df = pd.DataFrame({
        "SNS_A1": [1.0, 2.0, 3.0],
        "SNS_B1": [0, 1, 0],
        "some_other_column": ["x", "y", "z"],
        "claim": [0, 1, 0],
    })
    result = validate_upload(df, valid_config, label_col="claim")
    assert "some_other_column" in result.extra_columns


def test_type_mismatch_warning(valid_config):
    df = pd.DataFrame({
        "SNS_A1": ["not", "a", "number"],  # registered as scalar, but non-numeric
        "SNS_B1": [0, 1, 0],
        "claim": [0, 1, 0],
    })
    result = validate_upload(df, valid_config, label_col="claim")
    assert any("SNS_A1" in w for w in result.type_warnings)


def test_preprocess_report_is_attached(valid_config):
    df = pd.DataFrame({
        "SNS_A1": [1.0, 2.0, 3.0],
        "SNS_B1": [0, 1, 0],
        "claim": [0, 1, 0],
    })
    result = validate_upload(df, valid_config, label_col="claim")
    assert result.preprocess_report
    assert result.cleaned_df is not None
    assert result.raw_df is not None
