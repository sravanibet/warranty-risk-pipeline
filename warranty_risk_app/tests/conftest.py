import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def valid_config():
    return {
        "plant_id": "TEST_PLANT",
        "plant_name": "Test Plant",
        "stations": [
            {
                "station_uid": "STN_A",
                "station_name": "Station A",
                "station_order": 1,
                "station_type": "assembly",
                "connected_to": ["STN_B"],
                "sensors": [
                    {
                        "sensor_uid": "SNS_A1",
                        "sensor_name": "Sensor A1",
                        "data_type": "scalar",
                        "lower_limit": 0, "upper_limit": 10,
                        "risk_weight": 1.5,
                    }
                ],
            },
            {
                "station_uid": "STN_B",
                "station_name": "Station B",
                "station_order": 2,
                "station_type": "inspection",
                "connected_to": [],
                "sensors": [
                    {
                        "sensor_uid": "SNS_B1",
                        "sensor_name": "Sensor B1",
                        "data_type": "binary",
                        "risk_weight": "excluded",
                    }
                ],
            },
        ],
    }


@pytest.fixture
def synthetic_feature_df():
    # A small, deterministic feature matrix with a known warranty_claim label.
    rng = np.random.RandomState(42)
    n = 60
    sns_a1 = rng.normal(5, 2, n)
    sns_b1 = rng.binomial(1, 0.2, n)
    # Claims correlate strongly with high SNS_A1 and SNS_B1 == 1 by construction
    risk = (sns_a1 > 7).astype(int) | (sns_b1 == 1)
    df = pd.DataFrame({
        "SNS_A1": sns_a1,
        "SNS_B1": sns_b1,
        "warranty_claim": risk,
    })
    return df


@pytest.fixture
def synthetic_manufacturing_readings():
    # Raw sensor readings for one vehicle, shaped like manufacturing_df.
    return pd.DataFrame([
        {"serial_number": "SN00001", "station_uid": "STN_A", "sensor_uid": "SNS_A1",
         "sensor_type": "scalar", "timestamp": pd.Timestamp("2026-01-01T00:00:00"), "value": 4.2},
        {"serial_number": "SN00001", "station_uid": "STN_B", "sensor_uid": "SNS_B1",
         "sensor_type": "binary", "timestamp": pd.Timestamp("2026-01-01T00:00:05"), "value": 0},
    ])
