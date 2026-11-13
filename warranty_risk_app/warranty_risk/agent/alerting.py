# Alerting stage of the IntelliMake Quality Monitoring Agent — publishes the
# Detection stage's output back to a dedicated MQTT topic, so results are
# themselves a consumable stream (per IntelliMake_Architecture.pdf section 2).
import json
import logging
import os

try:
    import paho.mqtt.client as mqtt
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


class MQTTAlerter:
    def __init__(self, config: dict = None):
        cfg = config or {}
        self.host       = cfg.get("host") or os.environ.get("MQTT_HOST", "localhost")
        self.port       = int(cfg.get("port") or os.environ.get("MQTT_PORT", 1883))
        self.username   = cfg.get("username") or os.environ.get("MQTT_USERNAME")
        self.password   = cfg.get("password") or os.environ.get("MQTT_PASSWORD")
        self.use_tls    = bool(cfg.get("use_tls") if cfg.get("use_tls") is not None
                                else os.environ.get("MQTT_USE_TLS", "").lower() == "true")
        self.topic_pub  = cfg.get("topic_pub") or os.environ.get("MQTT_TOPIC_PUB", "quality/alerts")

        self.enabled = _SDK_AVAILABLE
        self.connected = False
        self._client = None

    def connect(self) -> bool:
        if not self.enabled:
            logger.warning("paho-mqtt is not installed — alerting disabled (results won't be published).")
            return False
        try:
            self._client = mqtt.Client()
            if self.username:
                self._client.username_pw_set(self.username, self.password)
            if self.use_tls:
                self._client.tls_set()
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()
            self.connected = True
            return True
        except Exception as e:
            logger.warning("MQTT alerter could not connect to %s:%s — %s", self.host, self.port, e)
            self.connected = False
            return False

    def publish_operate_result(self, operate_result) -> bool:
        # Publishes an OperateResult (from orchestrator.score_vehicle_from_readings) as JSON.
        payload = {
            "vehicle_id": operate_result.vehicle_id,
            "probability": operate_result.probability,
            "recommendation": operate_result.recommendation,
            "narration": operate_result.narration,
            "top_features": (
                operate_result.lime_result["lime_df"][["Feature", "Weight"]].head(5).to_dict("records")
                if operate_result.lime_result else []
            ),
        }
        return self.publish(payload)

    def publish(self, payload: dict) -> bool:
        if not self.enabled or not self.connected:
            logger.info("Alert not published (alerter disabled/not connected): %s", payload)
            return False
        try:
            self._client.publish(self.topic_pub, json.dumps(payload), qos=1)
            return True
        except Exception:
            logger.exception("Failed to publish alert to %s", self.topic_pub)
            return False

    def stop(self):
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
