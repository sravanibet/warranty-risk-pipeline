import pandas as pd
from sklearn.preprocessing import LabelEncoder


def build_feature_matrix(manufacturing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw sensor records into one row per vehicle.
    Categorical sensor encoded with LabelEncoder; binary cast to int.
    Returns a (n_vehicles × n_sensors) DataFrame indexed by serial_number.
    """
    df = manufacturing_df.copy()

    cat_mask = df["sensor_type"] == "categorical"
    if cat_mask.any():
        le = LabelEncoder()
        df.loc[cat_mask, "value"] = le.fit_transform(df.loc[cat_mask, "value"])

    bin_mask = df["sensor_type"] == "binary"
    df.loc[bin_mask, "value"] = df.loc[bin_mask, "value"].astype(int)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    feature_df = (
        df.groupby(["serial_number", "sensor_uid"])["value"]
        .mean()
        .unstack()
        .fillna(0)
    )
    return feature_df
