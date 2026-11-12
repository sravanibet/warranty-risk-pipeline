"""
Shared value encoding for sensor readings.

Training and live scoring must turn categorical sensor values into the SAME
integer codes. Fitting a fresh encoder per batch (or per single vehicle) gives
different codes depending on which categories happen to be present, so the
category map is fitted once on training data and reused everywhere.
"""
import pandas as pd

UNSEEN_CODE = -1


def fit_category_map(manufacturing_df: pd.DataFrame) -> dict:
    """Sorted-unique map {category_string: code} over all categorical readings."""
    cat = manufacturing_df.loc[manufacturing_df["sensor_type"] == "categorical", "value"]
    return {v: i for i, v in enumerate(sorted(set(cat.astype(str))))}


def encode_values(df: pd.DataFrame, category_map: dict = None) -> pd.DataFrame:
    """
    Return a copy of df with a numeric 'value' column: categorical readings
    mapped through category_map (fitted on df itself if None; categories not in
    the map become UNSEEN_CODE), binary cast to int, everything else coerced to
    numeric (unparseable values become NaN; callers decide how to fill).
    """
    df = df.copy()
    if category_map is None:
        category_map = fit_category_map(df)

    cat_mask = df["sensor_type"] == "categorical"
    values = df["value"].astype(object)
    if cat_mask.any():
        values[cat_mask] = values[cat_mask].astype(str).map(category_map).fillna(UNSEEN_CODE)

    bin_mask = df["sensor_type"] == "binary"
    if bin_mask.any():
        values[bin_mask] = values[bin_mask].astype(int)

    df["value"] = pd.to_numeric(values, errors="coerce")
    return df
