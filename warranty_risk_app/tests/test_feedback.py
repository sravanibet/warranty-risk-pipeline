import copy

import numpy as np
import pandas as pd

from warranty_risk.agent.feedback import (
    compute_weight_adjustments, apply_weight_adjustments, _explicitly_excluded, _current_weights,
)


def _fake_ga_result(sensors_opt, lowers, uppers):
    return {"success": True, "sensors_opt": sensors_opt, "lowers_opt": lowers, "uppers_opt": uppers}


def test_sensor_strongly_correlated_with_claims_gets_weight_increased(valid_config):
    # SNS_A1 out-of-tolerance (above upper bound of 5) should correlate with claim==1
    n = 200
    rng = np.random.RandomState(0)
    sns_a1 = rng.normal(5, 2, n)
    claim = (sns_a1 > 7).astype(int)
    feature_df = pd.DataFrame({"SNS_A1": sns_a1, "warranty_claim": claim})

    ga_result = _fake_ga_result(["sensor_1"], [0], [7])
    sensor_rename_inv = {"sensor_1": "SNS_A1"}

    adjustments = compute_weight_adjustments(feature_df, ga_result, sensor_rename_inv, valid_config)
    assert len(adjustments) == 1
    assert adjustments[0].sensor_uid == "SNS_A1"
    assert adjustments[0].new_weight > adjustments[0].old_weight  # positive correlation -> weight goes up


def test_excluded_sensor_is_never_adjusted(valid_config):
    n = 100
    rng = np.random.RandomState(1)
    sns_b1 = rng.binomial(1, 0.3, n)
    claim = sns_b1  # perfectly correlated
    feature_df = pd.DataFrame({"SNS_B1": sns_b1, "warranty_claim": claim})

    ga_result = _fake_ga_result(["sensor_1"], [0], [0])
    sensor_rename_inv = {"sensor_1": "SNS_B1"}  # SNS_B1 is "excluded" in valid_config

    adjustments = compute_weight_adjustments(feature_df, ga_result, sensor_rename_inv, valid_config)
    assert adjustments == []


def test_no_adjustments_when_ga_did_not_succeed(valid_config, synthetic_feature_df):
    ga_result = {"success": False}
    adjustments = compute_weight_adjustments(synthetic_feature_df, ga_result, {}, valid_config)
    assert adjustments == []


def test_no_adjustments_without_label_column(valid_config):
    df = pd.DataFrame({"SNS_A1": [1, 2, 3]})
    ga_result = _fake_ga_result(["sensor_1"], [0], [5])
    adjustments = compute_weight_adjustments(df, ga_result, {"sensor_1": "SNS_A1"}, valid_config)
    assert adjustments == []


def test_apply_weight_adjustments_writes_back_to_config(valid_config):
    original = copy.deepcopy(valid_config)
    n = 150
    rng = np.random.RandomState(2)
    sns_a1 = rng.normal(5, 2, n)
    claim = (sns_a1 > 7).astype(int)
    feature_df = pd.DataFrame({"SNS_A1": sns_a1, "warranty_claim": claim})
    ga_result = _fake_ga_result(["sensor_1"], [0], [7])

    adjustments = compute_weight_adjustments(feature_df, ga_result, {"sensor_1": "SNS_A1"}, valid_config)
    apply_weight_adjustments(valid_config, adjustments)

    new_weight = valid_config["stations"][0]["sensors"][0]["risk_weight"]
    old_weight = original["stations"][0]["sensors"][0]["risk_weight"]
    assert new_weight != old_weight


def test_weight_clipped_to_bounds(valid_config):
    # Force an extreme correlation with a large step to check clipping behavior
    n = 100
    sns_a1 = np.concatenate([np.full(50, 10.0), np.full(50, 0.0)])
    claim = np.concatenate([np.ones(50), np.zeros(50)]).astype(int)
    feature_df = pd.DataFrame({"SNS_A1": sns_a1, "warranty_claim": claim})
    ga_result = _fake_ga_result(["sensor_1"], [0], [5])

    adjustments = compute_weight_adjustments(
        feature_df, ga_result, {"sensor_1": "SNS_A1"}, valid_config, step=100.0
    )
    assert 0.1 <= adjustments[0].new_weight <= 5.0


def test_current_weights_reads_config_correctly(valid_config):
    weights = _current_weights(valid_config)
    assert weights["SNS_A1"] == 1.5
    assert weights["SNS_B1"] == 0.0


def test_explicitly_excluded_identifies_correct_sensors(valid_config):
    excluded = _explicitly_excluded(valid_config)
    assert excluded == {"SNS_B1"}
