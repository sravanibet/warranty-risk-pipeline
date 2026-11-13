import numpy as np
import pandas as pd
from ..preprocessing.encoding import encode_values


def _column_names(sensor_uids: list, max_seq_len: int) -> list:
    cols = []
    for s in sensor_uids:
        for t in range(max_seq_len):
            cols += [f"{s}_t{t}", f"{s}_mask_t{t}"]
    return cols


def _pad(df: pd.DataFrame, vehicle_ids: list, sensor_uids: list, max_seq_len: int) -> pd.DataFrame:
    # Vectorised core shared by training and live scoring: place each reading at
    # (vehicle row, sensor, time-step) using a per-(vehicle, sensor) counter, with
    # a 1 in the mask column for real readings and 0 left for padding.
    # df must already have a numeric 'value' column.
    df = df.sort_values(["serial_number", "sensor_uid", "timestamp"]).copy()
    df["_t"] = df.groupby(["serial_number", "sensor_uid"]).cumcount()
    df = df[df["_t"] < max_seq_len]

    v_idx = df["serial_number"].map({v: i for i, v in enumerate(vehicle_ids)})
    s_idx = df["sensor_uid"].map({s: i for i, s in enumerate(sensor_uids)})
    known = v_idx.notna() & s_idx.notna()   # sensors unseen at training time are ignored
    v_idx = v_idx[known].astype(int).to_numpy()
    s_idx = s_idx[known].astype(int).to_numpy()
    base = (s_idx * max_seq_len + df.loc[known, "_t"].to_numpy()) * 2

    arr = np.zeros((len(vehicle_ids), len(sensor_uids) * max_seq_len * 2))
    arr[v_idx, base] = df.loc[known, "value"].to_numpy()
    arr[v_idx, base + 1] = 1

    result = pd.DataFrame(arr, index=vehicle_ids, columns=_column_names(sensor_uids, max_seq_len))
    result.index.name = "serial_number"
    return result


def pad_and_mask(manufacturing_df: pd.DataFrame, category_map: dict = None) -> pd.DataFrame:
    # Encode categorical/binary sensor values and pad variable-length time
    # series to a uniform length per (vehicle, sensor) pair.
    #
    # Each sensor gets two columns per time step:
    #     <sensor_uid>_t<i>       — sensor value (0 for padding)
    #     <sensor_uid>_mask_t<i>  — 1 if real measurement, 0 if padding
    #
    # category_map: {category: code}; fitted on this data if omitted.
    # Returns a wide DataFrame indexed by serial_number with shape
    # (n_vehicles, n_sensors × max_seq_len × 2).
    df = encode_values(manufacturing_df, category_map)
    df["value"] = df["value"].fillna(0)
    sensor_uids, max_seq_len = compute_padding_meta(manufacturing_df)
    vehicle_ids = sorted(df["serial_number"].unique())
    return _pad(df, vehicle_ids, sensor_uids, max_seq_len)


def compute_padding_meta(manufacturing_df: pd.DataFrame) -> tuple:
    # Returns (sensor_uids, max_seq_len) exactly as pad_and_mask would derive
    # them from this DataFrame. Call this once at training time and reuse the
    # result so a single new vehicle scored later gets the identical column
    # layout the model was trained on.
    sensor_uids = sorted(manufacturing_df["sensor_uid"].unique())
    seq_lengths = manufacturing_df.groupby(["serial_number", "sensor_uid"]).size()
    max_seq_len = int(seq_lengths.max())
    return sensor_uids, max_seq_len


def pad_single_vehicle(readings_df: pd.DataFrame, sensor_uids: list, max_seq_len: int,
                       category_map: dict = None) -> pd.DataFrame:
    # Pad one new vehicle's raw sensor readings using a FIXED (sensor_uids,
    # max_seq_len, category_map) taken from training time, so the resulting
    # single-row DataFrame has exactly the columns and codes the model expects.
    # Pass the training-time category_map: without it, categories are encoded
    # from this single vehicle and will not match the trained codes.
    #
    # readings_df must contain rows for exactly one serial_number.
    vehicle_ids = readings_df["serial_number"].unique()
    if len(vehicle_ids) != 1:
        raise ValueError(f"Expected readings for exactly one vehicle, got {len(vehicle_ids)}.")

    df = encode_values(readings_df, category_map)
    df["value"] = df["value"].fillna(0)
    return _pad(df, list(vehicle_ids), sensor_uids, max_seq_len)


def build_padded_matrix(
    manufacturing_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    category_map: dict = None,
) -> pd.DataFrame:
    # Combine padded time-series features with aggregate features
    # (warranty_claim, within_tolerance_all, out_of_tolerance_count).
    #
    # Returns ml_data aligned by serial_number.
    padded = pad_and_mask(manufacturing_df, category_map)

    label_cols = [c for c in ["warranty_claim", "within_tolerance_all",
                               "out_of_tolerance_count"] if c in feature_df.columns]
    ml_data = padded.join(feature_df[label_cols], how="left")
    ml_data[label_cols] = ml_data[label_cols].fillna(0)
    return ml_data
