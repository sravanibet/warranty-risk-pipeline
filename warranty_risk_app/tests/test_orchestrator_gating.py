import pytest

from warranty_risk.agent import QualityMonitoringAgent


def test_cannot_provide_data_before_setup_confirmed():
    agent = QualityMonitoringAgent()
    agent.setup()  # not confirmed yet
    with pytest.raises(RuntimeError):
        agent.data("simulate", n_vehicles=10)


def test_cannot_confirm_setup_with_invalid_config():
    agent = QualityMonitoringAgent()
    agent.setup({"plant_id": "X"})  # missing required fields
    with pytest.raises(RuntimeError):
        agent.confirm_setup()


def test_cannot_learn_before_data():
    agent = QualityMonitoringAgent()
    agent.setup()
    agent.confirm_setup()
    with pytest.raises(RuntimeError):
        agent.learn()


def test_cannot_validate_before_learn():
    agent = QualityMonitoringAgent()
    agent.setup()
    agent.confirm_setup()
    pytest.importorskip("plant_simulator")
    agent.data("simulate", n_vehicles=20)
    with pytest.raises(RuntimeError):
        agent.validate()


def test_setup_with_valid_config_allows_confirm(valid_config):
    agent = QualityMonitoringAgent()
    result = agent.setup(valid_config)
    assert result.validation.is_valid
    agent.confirm_setup()
    assert agent.setup_confirmed is True


def test_upload_with_missing_sensors_does_not_unlock_data_phase(valid_config):
    import pandas as pd
    agent = QualityMonitoringAgent()
    agent.setup(valid_config)
    agent.confirm_setup()
    incomplete_df = pd.DataFrame({"SNS_A1": [1, 2, 3], "claim": [0, 1, 0]})  # missing SNS_B1
    result = agent.data("upload", df=incomplete_df, label_col="claim")
    assert not result.upload_validation.is_valid
