#
#Validates a plant configuration dict against the shape the pipeline needs.
#Collects every problem in one pass instead of stopping at the first one, so
#the agent can report everything wrong with a config at once.


from dataclasses import dataclass, field

REQUIRED_TOP_LEVEL = ["plant_id", "plant_name", "stations"]
REQUIRED_STATION_FIELDS = ["station_uid", "station_name", "station_type", "sensors"]
REQUIRED_SENSOR_FIELDS = ["sensor_uid", "data_type"]
KNOWN_DATA_TYPES = {"time_series", "scalar", "binary", "categorical"}


@dataclass
class ValidationResult:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def validate_config(config: dict) -> ValidationResult:
    result = ValidationResult()

    if not isinstance(config, dict):
        result.errors.append("Config must be a JSON object.")
        return result

    for key in REQUIRED_TOP_LEVEL:
        if key not in config:
            result.errors.append(f"Missing required top-level field: '{key}'")

    stations = config.get("stations")
    if stations is not None and not isinstance(stations, list):
        result.errors.append("'stations' must be a list.")
        stations = []
    stations = stations or []

    if not stations:
        result.errors.append("Config must define at least one station.")

    station_uids = set()
    for i, station in enumerate(stations):
        label = f"stations[{i}]"
        if not isinstance(station, dict):
            result.errors.append(f"{label} must be an object.")
            continue

        for field_name in REQUIRED_STATION_FIELDS:
            if field_name not in station:
                result.errors.append(f"{label} missing required field: '{field_name}'")

        uid = station.get("station_uid")
        if uid:
            if uid in station_uids:
                result.errors.append(f"Duplicate station_uid: '{uid}'")
            station_uids.add(uid)

        sensors = station.get("sensors")
        if sensors is not None and not isinstance(sensors, list):
            result.errors.append(f"{label}.sensors must be a list.")
            sensors = []
        if uid and not sensors:
            result.warnings.append(f"Station '{uid}' has no sensors defined.")

    # Second pass: connected_to references and sensor field checks
    all_sensor_uids = set()
    for i, station in enumerate(stations):
        if not isinstance(station, dict):
            continue
        label = station.get("station_uid", f"stations[{i}]")

        for target in station.get("connected_to", []) or []:
            if target not in station_uids:
                result.errors.append(
                    f"Station '{label}' connects to unknown station_uid: '{target}'"
                )

        for j, sensor in enumerate(station.get("sensors", []) or []):
            sensor_label = f"{label}.sensors[{j}]"
            if not isinstance(sensor, dict):
                result.errors.append(f"{sensor_label} must be an object.")
                continue

            for field_name in REQUIRED_SENSOR_FIELDS:
                if field_name not in sensor:
                    result.errors.append(f"{sensor_label} missing required field: '{field_name}'")

            uid = sensor.get("sensor_uid")
            if uid:
                if uid in all_sensor_uids:
                    result.errors.append(f"Duplicate sensor_uid: '{uid}'")
                all_sensor_uids.add(uid)

            dtype = sensor.get("data_type")
            if dtype and dtype not in KNOWN_DATA_TYPES:
                result.errors.append(
                    f"{sensor_label} has unknown data_type '{dtype}' "
                    f"(expected one of {sorted(KNOWN_DATA_TYPES)})"
                )

            weight = sensor.get("risk_weight")
            if weight is not None and weight != "excluded":
                try:
                    float(weight)
                except (TypeError, ValueError):
                    result.errors.append(
                        f"{sensor_label} has invalid risk_weight '{weight}' "
                        "(must be a number or \"excluded\")"
                    )

    if stations and not any((s.get("connected_to") or []) == [] for s in stations if isinstance(s, dict)):
        result.warnings.append(
            "No station has an empty connected_to — the line has no defined end point."
        )

    return result
