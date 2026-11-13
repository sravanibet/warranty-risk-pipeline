# Standalone Quality Monitoring Agent process — Listener -> Detection -> Alerting,
# matching IntelliMake_Architecture.pdf section 2.
#
# This is NOT the Streamlit app (that's a human-facing dashboard for walking
# through Setup/Data/Learning/Operate by hand). This script is the long-running
# process that would actually sit behind EMQX in production: it listens for
# live sensor readings, scores each finished vehicle against an already-trained
# model, and publishes the result back out as an alert.
#
# Run:
#     python stream_quality_agent.py
#
# Configuration (environment variables — see agent/mqtt_listener.py and
# agent/alerting.py for the full list):
#     MQTT_HOST, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD, MQTT_USE_TLS,
#     MQTT_TOPIC_SUB (default "quality/+/events"),
#     MQTT_TOPIC_PUB (default "quality/alerts")
#
# Until real EMQX credentials and a confirmed message schema are available
# (see plan.md / IntelliMake_Architecture.pdf open items), this defaults to
# localhost:1883 so it can be exercised against a local test broker — see
# mock_publisher.py in this same directory to simulate incoming readings.
#
# Also note: there is currently no persistence for a trained model, so this
# script trains one from simulated data on startup before it starts listening.
# Once real training data/a model registry exists, replace _bootstrap_model()
# with loading a previously trained pipeline instead of retraining every run.
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from warranty_risk.agent import QualityMonitoringAgent
from warranty_risk.agent.mqtt_listener import MQTTListener
from warranty_risk.agent.alerting import MQTTAlerter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("stream_quality_agent")


def _bootstrap_model(agent: QualityMonitoringAgent):
    # Placeholder until a trained model can be loaded from storage: registers
    # the default facility and trains once on simulated data so the Detection
    # stage has a model ready before the listener starts.
    logger.info("Bootstrapping: registering facility and training on simulated data...")
    agent.setup()
    agent.confirm_setup()
    agent.data("simulate", n_vehicles=300)
    learn_result = agent.learn(n_generations=20, pop_size=60)
    logger.info("Bootstrap complete: %s", learn_result.narration)


def main():
    agent = QualityMonitoringAgent()
    _bootstrap_model(agent)

    expected_sensor_uids = agent.pipe.trained_sensor_uids
    alerter = MQTTAlerter()
    alerter.connect()

    def on_window_complete(readings_df):
        vehicle_id = str(readings_df["serial_number"].iloc[0])
        logger.info("Window complete for vehicle %s — scoring...", vehicle_id)
        try:
            result = agent.score_vehicle_from_readings(readings_df)
        except Exception:
            logger.exception("Failed to score vehicle %s", vehicle_id)
            return
        logger.info("Result: %s", result.narration)
        alerter.publish_operate_result(result)

    listener = MQTTListener(expected_sensor_uids, on_window_complete)
    if not listener.connect():
        logger.error(
            "Could not connect the listener to an MQTT broker (host=%s port=%s). "
            "Set MQTT_HOST/MQTT_PORT env vars to point at a broker, or run a local "
            "test broker (e.g. Mosquitto) for development.",
            listener.host, listener.port,
        )
        return

    logger.info("Listening on %s for vehicle windows...", listener.topic_sub)
    try:
        listener.loop_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        listener.stop()
        alerter.stop()


if __name__ == "__main__":
    main()
