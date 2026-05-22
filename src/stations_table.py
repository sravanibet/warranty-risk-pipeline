# stations_table.py to initiate the stations and their connections, the functions are run from generate_stations.py
# to change any plant configuration, 1) update in this file stations_table dataframe, 
# 2) then run generate_stations.py to update plant_config.json file 
# 3) run 03_data_generation.ipynb to generate data based on the plant_config.json
from pathlib import Path
import pandas as pd
import json

def get_stations_table():
    """Create the station table and automatically assign station_order."""
    stations_table = pd.DataFrame([
        {"station_name": "Body Assembly", "station_uid": "STN_BODY", "station_type": "assembly", "capacity": 55, "connected_to": ["STN_CHASSIS"]},
        {"station_name": "Paint Shop", "station_uid": "STN_PAINT", "station_type": "coating", "capacity": 50, "connected_to": ["STN_CHASSIS"]},
        {"station_name": "Chassis Assembly", "station_uid": "STN_CHASSIS", "station_type": "assembly", "capacity": 60, "connected_to": ["STN_ELECTRICAL","STN_FINAL"]},
        {"station_name": "Electrical Install", "station_uid": "STN_ELECTRICAL", "station_type": "electrical", "capacity": 65, "connected_to": ["STN_FINAL"]},
        {"station_name": "Final Inspection", "station_uid": "STN_FINAL", "station_type": "inspection", "capacity": 70, "connected_to": ["STN_E2E"]},
        {"station_name": "End-to-End Test", "station_uid": "STN_E2E", "station_type": "testing", "capacity": 45, "connected_to": []},
    ])
    stations_table["station_order"] = range(1, len(stations_table)+1)
    return stations_table

def generate_plant_config(stations_table):

    # ------------------------------------------------
    # 1️⃣ Get project root (Warranty_risk_prototype)
    # ------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # ------------------------------------------------
    # 2️⃣ Define config folder
    # ------------------------------------------------
    CONFIG_DIR = PROJECT_ROOT / "data" / "config"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    output_path = CONFIG_DIR / "plant_config.json"

    # ------------------------------------------------
    # 3️⃣ Build plant config
    # ------------------------------------------------
    plant_config = {
        "plant_id": "AUTO_PLANT_001",
        "plant_name": "Automotive Final Assembly Plant",
        "stations": []
    }

    for idx, row in stations_table.iterrows():

        station_uid = row["station_uid"]

        # ------------------------------------------------
        # 4️⃣ Define sensors PROPERLY
        # ------------------------------------------------

        if station_uid == "STN_BODY":
            sensors = [
                {
                    "sensor_uid": "SNS_WELD_TEMP_001",
                    "sensor_name": "Welding Temperature",
                    "sensor_category": "thermal",
                    "data_type": "time_series",
                    "unit": "Celsius",
                    "lower_limit": 450,
                    "upper_limit": 520,
                    "sampling_hz": 2
                }
            ]

        elif station_uid == "STN_PAINT":
            sensors = [
                {
                    "sensor_uid": "SNS_PAINT_THICK_001",
                    "sensor_name": "Paint Thickness",
                    "sensor_category": "optical",
                    "data_type": "scalar",
                    "unit": "micron",
                    "lower_limit": 80,
                    "upper_limit": 120
                }
            ]

        elif station_uid == "STN_CHASSIS":
            sensors = [
                {
                    "sensor_uid": "SNS_TORQUE_001",
                    "sensor_name": "Wheel Torque",
                    "sensor_category": "mechanical",
                    "data_type": "time_series",
                    "unit": "Nm",
                    "lower_limit": 30,
                    "upper_limit": 50,
                    "sampling_hz": 5
                }
            ]

        elif station_uid == "STN_ELECTRICAL":
            sensors = [
                {
                    "sensor_uid": "SNS_CONTINUITY_001",
                    "sensor_name": "Wiring Continuity",
                    "sensor_category": "electrical",
                    "data_type": "binary"
                }
            ]

        elif station_uid == "STN_FINAL":
            sensors = [
                {
                    "sensor_uid": "SNS_VISUAL_DEFECT_001",
                    "sensor_name": "Visual Defect Detection",
                    "sensor_category": "vision",
                    "data_type": "categorical"
                }
            ]

        elif station_uid == "STN_E2E":
            sensors = [
                {
                    "sensor_uid": "SNS_VIBRATION_001",
                    "sensor_name": "Vehicle Vibration",
                    "sensor_category": "dynamic",
                    "data_type": "time_series",
                    "unit": "mm/s",
                    "lower_limit": 0,
                    "upper_limit": 4,
                    "sampling_hz": 10
                }
            ]

        else:
            sensors = []

        # ------------------------------------------------
        # 5️⃣ Build station dict
        # ------------------------------------------------

        station_dict = {
            "station_uid": station_uid,
            "station_name": row["station_name"],
            "station_order": row["station_order"],
            "station_type": row["station_type"],
            "capacity_per_hour": row["capacity"],
            "connected_to": row["connected_to"],
            "sensors": sensors
        }

        plant_config["stations"].append(station_dict)

    # ------------------------------------------------
    # 6️⃣ Save JSON
    # ------------------------------------------------
    with open(output_path, "w") as f:
        json.dump(plant_config, f, indent=2)

    print(f"\n✅ plant_config.json saved to:\n{output_path}\n")

    return plant_config