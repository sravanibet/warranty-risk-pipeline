"""
Listener stage of the IntelliMake Quality Monitoring Agent
(Listener -> Detection -> Alerting, per IntelliMake_Architecture.pdf).

Subscribes to a topic where sensor readings are published one at a time,
buffers them per vehicle (serial_number), and calls a callback with a
complete DataFrame of that vehicle's readings once every sensor registered
in the plant config has reported — the "sliding window" the architecture
doc describes, keyed by vehicle rather than by wall-clock time since a
vehicle's pass through the line is the natural window boundary here.

Expected message schema (until IntelliMake's actual schema is confirmed):
    topic:   quality/<station_uid>/events
    payload: JSON — {"serial_number", "station_uid", "sensor_uid",
                      "sensor_type", "timestamp", "value"}
This mirrors the row shape already used internally by manufacturing_df, so
a completed window can be handed directly to the existing detection code
(WarrantyRiskPipeline.score_new_vehicle) with no reshaping.

Connection is configured entirely from environment variables so no
credentials are hardcoded, and everything degrades to "disabled" rather
than crashing if paho-mqtt isn't installed or the broker is unreachable —
same pattern as narration.py's Gemini fallback.
"""
import json
import logging
import os
import threading
from datetime import datetime, timezone

import pandas as pd

try:
    import paho.mqtt.client as mqtt
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class MQTTListener:
    def __init__(self, expected_sensor_uids: list, on_window_complete, config: dict = None):
        """
        expected_sensor_uids : sensor_uids that must all be present before a
                                vehicle's window is considered complete
        on_window_complete   : callback(readings_df) called once per vehicle
        config               : optional overrides; falls back to env vars:
                                MQTT_HOST, MQTT_PORT, MQTT_USERNAME,
                                MQTT_PASSWORD, MQTT_USE_TLS, MQTT_TOPIC_SUB
        """
        cfg = config or {}
        self.host       = cfg.get("host") or os.environ.get("MQTT_HOST", "localhost")
        self.port       = int(cfg.get("port") or os.environ.get("MQTT_PORT", 1883))
        self.username   = cfg.get("username") or os.environ.get("MQTT_USERNAME")
        self.password    = cfg.get("password") or os.environ.get("MQTT_PASSWORD")
        self.use_tls    = bool(cfg.get("use_tls") if cfg.get("use_tls") is not None
                                else os.environ.get("MQTT_USE_TLS", "").lower() == "true")
        self.topic_sub  = cfg.get("topic_sub") or os.environ.get("MQTT_TOPIC_SUB", "quality/+/events")

        self.expected_sensor_uids = set(expected_sensor_uids)
        self.on_window_complete = on_window_complete

        self.enabled = _SDK_AVAILABLE
        self.connected = False
        self._client = None
        self._buffers = {}   # serial_number -> list[dict]
        self._lock = threading.Lock()

    def connect(self) -> bool:
        if not self.enabled:
            logger.warning("paho-mqtt is not installed — MQTT listener disabled.")
            return False
        try:
            self._client = mqtt.Client()
            if self.username:
                self._client.username_pw_set(self.username, self.password)
            if self.use_tls:
                self._client.tls_set()
            self._client.on_connect = self._on_connect
            self._client.on_message = self._on_message
            self._client.connect(self.host, self.port, keepalive=60)
            return True
        except Exception as e:
            logger.warning("MQTT listener could not connect to %s:%s — %s", self.host, self.port, e)
            self.connected = False
            return False

    def _on_connect(self, client, userdata, flags, rc):
        self.connected = (rc == 0)
        if self.connected:
            client.subscribe(self.topic_sub, qos=1)
            logger.info("MQTT listener subscribed to %s", self.topic_sub)
        else:
            logger.warning("MQTT connect failed, rc=%s", rc)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            logger.warning("Dropped unparseable MQTT message on %s", msg.topic)
            return

        required = {"serial_number", "station_uid", "sensor_uid", "sensor_type", "value"}
        if not required.issubset(payload):
            logger.warning("Dropped MQTT message missing required fields: %s", payload)
            return

        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        vid = payload["serial_number"]

        with self._lock:
            self._buffers.setdefault(vid, []).append(payload)
            seen_sensors = {r["sensor_uid"] for r in self._buffers[vid]}
            window_complete = self.expected_sensor_uids.issubset(seen_sensors)
            if window_complete:
                readings = self._buffers.pop(vid)

        if window_complete:
            readings_df = pd.DataFrame(readings)
            readings_df["timestamp"] = pd.to_datetime(readings_df["timestamp"])
            try:
                self.on_window_complete(readings_df)
            except Exception:
                logger.exception("on_window_complete callback failed for vehicle %s", vid)

    def loop_forever(self):
        if not self.connected or self._client is None:
            raise RuntimeError("MQTT listener is not connected — call connect() first.")
        self._client.loop_forever()

    def loop_start(self):
        if not self.connected or self._client is None:
            raise RuntimeError("MQTT listener is not connected — call connect() first.")
        self._client.loop_start()

    def stop(self):
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
