import pytest

from warranty_risk.agent import QualityMonitoringAgent


@pytest.fixture(scope="module")
def learned_agent():
    pytest.importorskip("plant_simulator")
    agent = QualityMonitoringAgent()
    agent.setup()
    agent.confirm_setup()
    agent.data("simulate", n_vehicles=100)
    agent.learn(n_generations=5, pop_size=15)
    return agent


def test_validate_returns_consistent_structure(learned_agent):
    result = learned_agent.validate(n_runs=3)
    assert result.n_runs == 3
    assert len(result.per_run) == 3
    assert 0.0 <= result.f1_mean <= 1.0
    assert result.f1_std >= 0.0


def test_validate_computes_a_real_baseline(learned_agent):
    result = learned_agent.validate(n_runs=3)
    assert 0.0 <= result.baseline_f1_mean <= 1.0


def test_lift_over_baseline_is_mean_difference(learned_agent):
    result = learned_agent.validate(n_runs=3)
    assert result.lift_over_baseline == pytest.approx(result.f1_mean - result.baseline_f1_mean)


def test_each_run_uses_a_different_seed_and_may_select_different_models(learned_agent):
    result = learned_agent.validate(n_runs=5)
    seeds = [r["seed"] for r in result.per_run]
    assert seeds == list(range(5))
