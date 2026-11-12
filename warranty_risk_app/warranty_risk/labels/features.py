import pandas as pd
from ..preprocessing.encoding import encode_values


def build_feature_matrix(manufacturing_df: pd.DataFrame, category_map: dict = None) -> pd.DataFrame:
    """
    Aggregate raw sensor records into one row per vehicle.
    Categorical sensor encoded with a shared category map (see preprocessing.encoding); binary cast to int.
    Returns a (n_vehicles × n_sensors) DataFrame indexed by serial_number.
    """
    df = encode_values(manufacturing_df, category_map)

    feature_df = (
        df.groupby(["serial_number", "sensor_uid"])["value"]
        .mean()
        .unstack()
        .fillna(0)
    )
    return feature_df
