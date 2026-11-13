# Publishes one simulated vehicle's sensor readings to MQTT, one message per
# reading — for testing stream_quality_agent.py without a real EMQX connection.
#
# Requires a local MQTT broker (e.g. Mosquitto: `mosquitto` with default
# config listens on localhost:1883) and the plant_simulator package.
#
# Run:
#     python mock_publisher.py
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from warranty_risk.config import load_config, get_stations

try:
    from plant_simulator import generate_vehicle_metadata, generate_manufacturing_data
except ImportError:
    print("plant_simulator is not installed — run: pip install -e ./plant_simulator")
    sys.exit(1)

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("paho-mqtt is not installed — run: pip install paho-mqtt")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock_publisher")

import os
HOST = os.environ.get("MQTT_HOST", "localhost")
PORT = int(os.environ.get("MQTT_PORT", 1883))
TOPIC_PREFIX = os.environ.get("MQTT_TOPIC_PREFIX", "quality")


def main():
    config = load_config()
    stations = get_stations(config)
    vehicle_df = generate_vehicle_metadata(n=1)
    readings = generate_manufacturing_data(vehicle_df, stations)

    client = mqtt.Client()
    try:
        client.connect(HOST, PORT, keepalive=60)
    except Exception as e:
        print(f"Could not connect to {HOST}:{PORT} — is a local MQTT broker running? ({e})")
        sys.exit(1)
    client.loop_start()

    vehicle_id = readings["serial_number"].iloc[0]
    logger.info("Publishing %d readings for vehicle %s", len(readings), vehicle_id)

    for _, row in readings.iterrows():
        payload = {
            "serial_number": row["serial_number"],
            "station_uid": row["station_uid"],
            "sensor_uid": row["sensor_uid"],
            "sensor_type": row["sensor_type"],
            "timestamp": row["timestamp"].isoformat(),
            "value": row["value"],
        }
        topic = f"{TOPIC_PREFIX}/{row['station_uid']}/events"
        client.publish(topic, json.dumps(payload), qos=1)
        time.sleep(0.05)

    logger.info("Done publishing vehicle %s.", vehicle_id)
    client.loop_stop()
    client.disconnect()


if __name__ == "__main__":
    main()
