import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_vehicle_metadata(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    start_time = datetime(2024, 1, 1, 8, 0, 0)
    rows = []
    for i in range(n):
        prod_time = start_time + timedelta(minutes=i * 5)
        sale_time = prod_time + timedelta(days=int(np.random.randint(30, 90)))
        rows.append({
            "serial_number":          f"SN{i:05d}",
            "production_timestamp":   prod_time,
            "sale_timestamp":         sale_time,
            "product_status_at_sale": np.random.choice(["new", "used", "refurbished"]),
            "shift_id":               i % 3,
            "env_stress":             int(np.random.choice([0, 1], p=[0.97, 0.03])),
        })
    return pd.DataFrame(rows)
