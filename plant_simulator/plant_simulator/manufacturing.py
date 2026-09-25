import numpy as np
import pandas as pd
from datetime import timedelta

PHI = 0.85  # AR(1) autocorrelation coefficient


def generate_manufacturing_data(vehicle_df: pd.DataFrame, stations: list) -> pd.DataFrame:
    """
    Simulate sensor readings for every vehicle through every station.
    Uses AR(1) for time-series/scalar sensors, probabilistic models for binary/categorical.
    Drift, environmental stress, and shift noise are applied per vehicle.
    """
    n_vehicles = len(vehicle_df)
    records = []

    for idx, vehicle in vehicle_df.iterrows():
        serial_number = vehicle["serial_number"]
        base_time     = vehicle["production_timestamp"]
        drift_factor  = idx / n_vehicles
        env_drift     = vehicle["env_stress"] * 0.20
        shift_noise   = 1.0 + (vehicle["shift_id"] == 2) * 0.10

        for station in stations:
            for sensor in station["sensors"]:
                data_type   = sensor.get("data_type", "scalar")
                sampling_hz = sensor.get("sampling_hz", 1)

                if data_type == "time_series":
                    duration   = int(np.random.randint(1, 4))
                    num_points = int(duration * sampling_hz)
                else:
                    num_points = 1

                if data_type not in ("binary", "categorical"):
                    lower     = sensor.get("lower_limit", 0)
                    upper     = sensor.get("upper_limit", 1)
                    nominal   = (lower + upper) / 2
                    std_dev   = (upper - lower) / 8 * shift_noise
                    drift     = (drift_factor + env_drift) * (upper - lower) * 0.10
                    overshoot = (upper - lower) * 0.05
                    prev = float(np.clip(np.random.normal(nominal + drift, std_dev), lower, upper))

                for m in range(num_points):
                    timestamp = base_time + timedelta(seconds=m / sampling_hz)

                    if data_type == "binary":
                        total_drift = min(drift_factor + env_drift, 1.0)
                        p_fail = 0.03 + total_drift * 0.12
                        value  = int(np.random.binomial(1, p_fail))

                    elif data_type == "categorical":
                        total_drift = min(drift_factor + env_drift, 1.0)
                        p_ok    = max(0.70, 0.90 - total_drift * 0.20)
                        p_minor = min(0.22, 0.08 + total_drift * 0.12)
                        p_major = max(0.001, 1.0 - p_ok - p_minor)
                        total   = p_ok + p_minor + p_major
                        value   = np.random.choice(
                            ["ok", "minor", "major"],
                            p=[p_ok / total, p_minor / total, p_major / total]
                        )

                    else:
                        noise = np.random.normal(0, std_dev * np.sqrt(1 - PHI ** 2))
                        raw   = PHI * prev + (1 - PHI) * (nominal + drift) + noise
                        value = float(np.clip(raw, lower, upper + overshoot))
                        prev  = value

                    records.append({
                        "serial_number": serial_number,
                        "station_uid":   station["station_uid"],
                        "sensor_uid":    sensor["sensor_uid"],
                        "sensor_type":   data_type,
                        "timestamp":     timestamp,
                        "value":         value,
                    })

    return pd.DataFrame(records)
