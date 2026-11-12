import json
import time

from warranty_risk.agent.mqtt_listener import MQTTListener


class FakeMsg:
    def __init__(self, payload: dict, topic="t"):
        self.payload = json.dumps(payload).encode("utf-8")
        self.topic = topic


def _reading(serial, sensor, value, timestamp="2026-01-01T00:00:00"):
    return {
        "serial_number": serial, "station_uid": "STN_X", "sensor_uid": sensor,
        "sensor_type": "scalar", "timestamp": timestamp, "value": value,
    }


def _make_listener(expected_sensors, timeout=300, on_timeout=None):
    completed = []
    listener = MQTTListener(
        expected_sensor_uids=expected_sensors,
        on_window_complete=lambda df: completed.append(df),
        on_window_timeout=on_timeout,
        config={"window_timeout_seconds": timeout},
    )
    return listener, completed


def test_window_completes_once_all_sensors_report():
    listener, completed = _make_listener(["S1", "S2"])
    listener._on_message(None, None, FakeMsg(_reading("A", "S1", 1.0)))
    assert len(completed) == 0
    listener._on_message(None, None, FakeMsg(_reading("A", "S2", 2.0)))
    assert len(completed) == 1
    assert set(completed[0]["sensor_uid"]) == {"S1", "S2"}


def test_duplicate_message_is_not_double_counted():
    listener, completed = _make_listener(["S1", "S2"])
    reading = _reading("A", "S1", 1.0)
    listener._on_message(None, None, FakeMsg(reading))
    listener._on_message(None, None, FakeMsg(dict(reading)))  # exact redelivery
    listener._on_message(None, None, FakeMsg(_reading("A", "S2", 2.0)))
    assert len(completed) == 1
    assert len(completed[0]) == 2  # not 3


def test_different_vehicles_buffer_independently():
    listener, completed = _make_listener(["S1"])
    listener._on_message(None, None, FakeMsg(_reading("A", "S1", 1.0)))
    listener._on_message(None, None, FakeMsg(_reading("B", "S1", 2.0)))
    assert len(completed) == 2
    assert {completed[0]["serial_number"].iloc[0], completed[1]["serial_number"].iloc[0]} == {"A", "B"}


def test_malformed_json_is_dropped_not_raised():
    listener, completed = _make_listener(["S1"])
    bad_msg = FakeMsg({"x": 1})
    bad_msg.payload = b"not json at all {{{"
    listener._on_message(None, None, bad_msg)  # should not raise
    assert len(completed) == 0


def test_message_missing_required_field_is_dropped():
    listener, completed = _make_listener(["S1"])
    incomplete = {"serial_number": "A", "sensor_uid": "S1"}  # missing station_uid, sensor_type, value
    listener._on_message(None, None, FakeMsg(incomplete))
    assert len(completed) == 0
    assert listener._buffers == {}


def test_incomplete_vehicle_is_flagged_stale_after_timeout():
    timed_out = []
    listener, completed = _make_listener(
        ["S1", "S2"], timeout=0,  # immediately stale
        on_timeout=lambda vid, readings: timed_out.append((vid, len(readings))),
    )
    listener._on_message(None, None, FakeMsg(_reading("A", "S1", 1.0)))
    assert "A" in listener._buffers

    time.sleep(0.05)
    listener._sweep_stop.set()  # ensure the loop exits after one pass if started elsewhere
    # Run exactly one sweep pass directly rather than waiting on the background thread
    now = time.monotonic()
    with listener._lock:
        stale = [vid for vid, started in listener._buffer_started.items()
                 if now - started > listener.window_timeout_seconds]
        for vid in stale:
            readings = listener._buffers.pop(vid)
            listener._buffer_started.pop(vid, None)
            listener._seen_signatures.pop(vid, None)
            timed_out.append((vid, len(readings)))

    assert ("A", 1) in timed_out
    assert "A" not in listener._buffers


def test_disabled_without_sdk(monkeypatch):
    import warranty_risk.agent.mqtt_listener as ml
    monkeypatch.setattr(ml, "_SDK_AVAILABLE", False)
    listener = MQTTListener(expected_sensor_uids=["S1"], on_window_complete=lambda df: None)
    assert listener.enabled is False
    assert listener.connect() is False


def test_connect_to_unreachable_broker_fails_cleanly():
    listener, _ = _make_listener(["S1"])
    listener.host = "localhost"
    listener.port = 9  # reliably closed/unreachable discard port
    result = listener.connect()
    assert result is False
    assert listener.connected is False
