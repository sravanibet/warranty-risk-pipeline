from warranty_risk.agent.config_validator import validate_config


def test_valid_config_passes(valid_config):
    result = validate_config(valid_config)
    assert result.is_valid
    assert result.errors == []


def test_missing_top_level_field_reported():
    result = validate_config({"plant_id": "X"})
    assert not result.is_valid
    assert any("plant_name" in e for e in result.errors)
    assert any("stations" in e for e in result.errors)


def test_empty_stations_list_is_invalid():
    result = validate_config({"plant_id": "X", "plant_name": "X", "stations": []})
    assert not result.is_valid
    assert any("at least one station" in e for e in result.errors)


def test_dangling_connected_to_reported(valid_config):
    valid_config["stations"][0]["connected_to"] = ["STN_GHOST"]
    result = validate_config(valid_config)
    assert not result.is_valid
    assert any("STN_GHOST" in e for e in result.errors)


def test_duplicate_station_uid_reported(valid_config):
    valid_config["stations"].append(dict(valid_config["stations"][0]))
    result = validate_config(valid_config)
    assert not result.is_valid
    assert any("Duplicate station_uid" in e for e in result.errors)


def test_duplicate_sensor_uid_reported(valid_config):
    valid_config["stations"][1]["sensors"].append(dict(valid_config["stations"][0]["sensors"][0]))
    result = validate_config(valid_config)
    assert not result.is_valid
    assert any("Duplicate sensor_uid" in e for e in result.errors)


def test_unknown_data_type_reported(valid_config):
    valid_config["stations"][0]["sensors"][0]["data_type"] = "not_a_real_type"
    result = validate_config(valid_config)
    assert not result.is_valid
    assert any("unknown data_type" in e for e in result.errors)


def test_invalid_risk_weight_reported(valid_config):
    valid_config["stations"][0]["sensors"][0]["risk_weight"] = "not_a_number"
    result = validate_config(valid_config)
    assert not result.is_valid
    assert any("invalid risk_weight" in e for e in result.errors)


def test_valid_excluded_risk_weight_is_not_an_error(valid_config):
    result = validate_config(valid_config)
    assert result.is_valid  # STN_B's sensor is "excluded", which is valid


def test_missing_required_sensor_field_reported(valid_config):
    del valid_config["stations"][0]["sensors"][0]["data_type"]
    result = validate_config(valid_config)
    assert not result.is_valid
    assert any("missing required field: 'data_type'" in e for e in result.errors)


def test_station_with_no_sensors_warns_not_errors(valid_config):
    valid_config["stations"][0]["sensors"] = []
    result = validate_config(valid_config)
    assert result.is_valid
    assert any("no sensors" in w for w in result.warnings)


def test_no_end_station_warns(valid_config):
    valid_config["stations"][1]["connected_to"] = ["STN_A"]  # now nothing points to an empty list
    result = validate_config(valid_config)
    assert any("no defined end point" in w for w in result.warnings)


def test_non_dict_config_is_invalid():
    result = validate_config("not a dict")
    assert not result.is_valid
