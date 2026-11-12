import numpy as np
import pandas as pd
import pytest

from warranty_risk.preprocessing import Preprocessor, auto_preprocess


def _df():
    return pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "b": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],     # constant
        "c": ["x", "y", "x", "y", "x", "y"],            # non-numeric
        "d": [np.nan, np.nan, np.nan, np.nan, 1.0, 2.0],  # >50% missing
        "e": [0.1, 0.4, 0.2, 0.9, 0.3, 0.8],
        "claim": [0, 1, 0, 1, 0, 1],
    })


def test_auto_preprocess_drops_and_imputes():
    out, report = auto_preprocess(_df(), label_col="claim")
    assert list(out.columns) == ["a", "e", "claim"]
    assert out["a"].isnull().sum() == 0
    assert out["a"].iloc[2] == 4.0                       # median of 1,2,4,5,6
    assert {r["step"] for r in report} >= {"Input shape", "Missing values", "Output shape"}


def test_imputation_uses_fit_data_only():
    train = _df()
    test = pd.DataFrame({"a": [np.nan], "e": [0.5], "claim": [1],
                         "b": [10.0], "c": ["x"], "d": [np.nan]})
    pp = Preprocessor(label_col="claim").fit(train)
    out = pp.transform(test)
    assert out["a"].iloc[0] == 4.0                       # train median, not test's own
    assert list(out.columns) == ["a", "e", "claim"]


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        Preprocessor().transform(_df())


def test_transform_missing_fitted_column_raises():
    pp = Preprocessor(label_col="claim").fit(_df())
    with pytest.raises(ValueError, match="a"):
        pp.transform(_df().drop(columns=["a"]))


def test_mostly_missing_label_is_not_dropped_as_a_column():
    df = _df()
    df["claim"] = [1, np.nan, np.nan, np.nan, np.nan, 0]
    out, report = auto_preprocess(df, label_col="claim")
    assert "claim" in out.columns and len(out) == 2
    assert any("label" in r["detail"] for r in report)


def test_yes_no_label_is_coerced():
    df = _df()
    df["claim"] = ["yes", "no", "no", "yes", "no", "yes"]
    out, _ = auto_preprocess(df, label_col="claim")
    assert out["claim"].tolist() == [1, 0, 0, 1, 0, 1]


def test_unreadable_label_warns_instead_of_crashing_in_one_shot():
    df = _df()
    df["claim"] = ["maybe"] * 6
    out, report = auto_preprocess(df, label_col="claim")
    assert any(r["status"] == "warning" and "Could not coerce" in r["detail"] for r in report)
    with pytest.raises(ValueError):
        Preprocessor(label_col="claim").fit(df).transform(df)


def test_live_rows_are_not_deduplicated():
    pp = Preprocessor(label_col="claim").fit(_df())
    row = _df().iloc[[0]]
    assert len(pp.transform(pd.concat([row, row]))) == 2
