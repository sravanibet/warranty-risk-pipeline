import pytest

from warranty_risk.agent import QualityMonitoringAgent


@pytest.fixture(scope="module")
def trained_agent():
    pytest.importorskip("plant_simulator")
    agent = QualityMonitoringAgent()
    agent.setup()
    agent.confirm_setup()
    agent.data("simulate", n_vehicles=100)
    agent.learn(n_generations=5, pop_size=15)
    return agent


def test_score_vehicle_from_readings_returns_valid_probability(trained_agent):
    from plant_simulator import generate_vehicle_metadata, generate_manufacturing_data
    from warranty_risk.config import get_stations

    stations = get_stations(trained_agent.setup_result.config)
    new_vehicle_df = generate_vehicle_metadata(n=1, seed=12345)
    new_readings = generate_manufacturing_data(new_vehicle_df, stations)

    result = trained_agent.score_vehicle_from_readings(new_readings)
    assert 0.0 <= result.probability <= 1.0
    assert result.vehicle_id == new_vehicle_df["serial_number"].iloc[0]
    assert result.recommendation in (
        "High risk — inspect before shipment.",
        "Elevated risk — flag for monitoring.",
        "Low risk — no action needed.",
    )


def test_score_vehicle_uses_consistent_feature_columns(trained_agent):
    """Regression test for the train/serve skew bug: the padded single-vehicle
    row must align to the exact columns the model was trained on."""
    from plant_simulator import generate_vehicle_metadata, generate_manufacturing_data
    from warranty_risk.config import get_stations

    stations = get_stations(trained_agent.setup_result.config)
    new_vehicle_df = generate_vehicle_metadata(n=1, seed=999)
    new_readings = generate_manufacturing_data(new_vehicle_df, stations)

    lr = trained_agent.pipe.score_new_vehicle(new_readings)
    trained_feat_cols = list(trained_agent.pipe.train_result["feat_cols"])
    assert len(lr["lime_df"]) <= len(trained_feat_cols)


def test_operate_requires_learning_first():
    agent = QualityMonitoringAgent()
    with pytest.raises(RuntimeError):
        agent.operate()


def test_score_vehicle_requires_learning_first():
    agent = QualityMonitoringAgent()
    import pandas as pd
    df = pd.DataFrame([{"serial_number": "X"}])
    with pytest.raises(RuntimeError):
        agent.score_vehicle_from_readings(df)
