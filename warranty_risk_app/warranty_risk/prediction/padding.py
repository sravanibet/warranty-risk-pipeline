import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def pad_and_mask(manufacturing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical/binary sensor values and pad variable-length time
    series to a uniform length per (vehicle, sensor) pair.

    Each sensor gets two columns per time step:
        <sensor_uid>_t<i>       — sensor value (0 for padding)
        <sensor_uid>_mask_t<i>  — 1 if real measurement, 0 if padding

    Returns a wide DataFrame indexed by serial_number with shape
    (n_vehicles, n_sensors × max_seq_len × 2).
    """
    df = manufacturing_df.copy()

    # Encode categorical
    cat_mask = df["sensor_type"] == "categorical"
    if cat_mask.any():
        le = LabelEncoder()
        df.loc[cat_mask, "value"] = le.fit_transform(df.loc[cat_mask, "value"])

    bin_mask = df["sensor_type"] == "binary"
    df.loc[bin_mask, "value"] = df.loc[bin_mask, "value"].astype(int)

    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0)

    df = df.sort_values(["serial_number", "sensor_uid", "timestamp"])

    seq_lengths = df.groupby(["serial_number", "sensor_uid"]).size()
    max_seq_len = int(seq_lengths.max())

    sensor_uids  = sorted(df["sensor_uid"].unique())
    vehicle_ids  = sorted(df["serial_number"].unique())

    # Build column names
    cols = []
    for s in sensor_uids:
        for t in range(max_seq_len):
            cols += [f"{s}_t{t}", f"{s}_mask_t{t}"]

    rows = {}
    for vid in vehicle_ids:
        row = np.zeros(len(cols))
        vdf = df[df["serial_number"] == vid]
        for s in sensor_uids:
            sdf = vdf[vdf["sensor_uid"] == s]["value"].values
            for t, v in enumerate(sdf[:max_seq_len]):
                base = (sensor_uids.index(s) * max_seq_len + t) * 2
                row[base]     = v
                row[base + 1] = 1  # real
        rows[vid] = row

    result = pd.DataFrame.from_dict(rows, orient="index", columns=cols)
    result.index.name = "serial_number"
    return result


def build_padded_matrix(
    manufacturing_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine padded time-series features with aggregate features
    (warranty_claim, within_tolerance_all, out_of_tolerance_count).

    Returns ml_data aligned by serial_number.
    """
    padded = pad_and_mask(manufacturing_df)

    label_cols = [c for c in ["warranty_claim", "within_tolerance_all",
                               "out_of_tolerance_count"] if c in feature_df.columns]
    ml_data = padded.join(feature_df[label_cols], how="left")
    ml_data[label_cols] = ml_data[label_cols].fillna(0)
    return ml_data
