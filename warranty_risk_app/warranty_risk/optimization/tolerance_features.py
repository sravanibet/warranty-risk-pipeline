import numpy as np
import pandas as pd


def compute_tolerance_features(
    feature_df: pd.DataFrame,
    lowers_opt: np.ndarray,
    uppers_opt: np.ndarray,
    sensors_opt: list,
    sensor_rename_inv: dict,
) -> pd.DataFrame:
    # Derive two binary/count features from GA-optimized bounds.
    #
    # For each vehicle, check whether each sensor mean falls within its
    # optimized [lower, upper] interval. Returns feature_df with two
    # new columns appended (does not mutate input).
    #
    #  Columns added:
    #     within_tolerance_all    — 1 if ALL sensors in bounds, else 0
    #     out_of_tolerance_count  — number of sensors outside bounds (0–N)
    df = feature_df.copy()

    real_names = [sensor_rename_inv[s] for s in sensors_opt]

    in_bounds = pd.DataFrame(index=df.index)
    for s, real, lo, hi in zip(sensors_opt, real_names, lowers_opt, uppers_opt):
        col = real if real in df.columns else s
        if col in df.columns:
            in_bounds[s] = df[col].between(lo, hi).astype(int)
        else:
            in_bounds[s] = 0

    df["within_tolerance_all"]   = (in_bounds.sum(axis=1) == len(sensors_opt)).astype(int)
    df["out_of_tolerance_count"] = (1 - in_bounds).sum(axis=1)

    return df
