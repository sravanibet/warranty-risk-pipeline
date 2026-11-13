# The evaluation must be leakage-free: no test vehicle's label may influence any
# fitted component (tolerance bounds, model choice, decision threshold).
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from warranty_risk.labels.sigmoid import generate_warranty_labels
from warranty_risk.prediction.models import split_vehicles, train_and_select


@pytest.fixture(scope="module")
def learned_pipe():
    pytest.importorskip("plant_simulator")
    from warranty_risk.agent import QualityMonitoringAgent
    agent = QualityMonitoringAgent()
    agent.setup()
    agent.confirm_setup()
    agent.data("simulate", n_vehicles=300)
    agent.learn(n_generations=5, pop_size=15)
    return agent.pipe


def test_split_is_disjoint_complete_and_stratified():
    y = pd.Series((np.random.RandomState(0).rand(200) < 0.15).astype(int),
                  index=[f"V{i}" for i in range(200)])
    s = split_vehicles(y, random_state=1)
    parts = [set(s["train"]), set(s["val"]), set(s["test"])]
    assert parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2])
    assert set().union(*parts) == set(y.index)
    rates = [y.loc[list(p)].mean() for p in parts]
    assert max(rates) - min(rates) < 0.05


def test_ga_only_ever_sees_train_vehicles(learned_pipe, monkeypatch):
    # Spy on what the optimiser receives; it must be exactly the train vehicles.
    import warranty_risk.pipeline as pl
    seen = {}
    real = pl.run_ga_optimization

    def spy(**kwargs):
        seen["ids"] = set(kwargs["feature_df"].index)
        seen["holdout"] = kwargs.get("internal_holdout")
        return real(**kwargs)

    monkeypatch.setattr(pl, "run_ga_optimization", spy)
    p = learned_pipe
    p.fit_tolerance(p.feature_df_pre_opt, p.split["train"])
    assert seen["ids"] == set(p.split["train"])
    assert not (seen["ids"] & set(p.split["val"])) and not (seen["ids"] & set(p.split["test"]))
    assert seen["holdout"] == 0


def test_model_and_threshold_do_not_depend_on_test_labels(learned_pipe):
    p = learned_pipe
    ml = p.ml_data.copy()
    a = train_and_select(ml, split=p.split)
    ml.loc[p.split["test"], "warranty_claim"] = 1 - ml.loc[p.split["test"], "warranty_claim"]
    b = train_and_select(ml, split=p.split)
    assert a["best_name"] == b["best_name"]
    assert a["best_threshold"] == b["best_threshold"]
    assert list(a["results_df"]["Val_F1"]) == list(b["results_df"]["Val_F1"])


def test_shuffled_labels_give_chance_level_auc(learned_pipe):
    ml = learned_pipe.ml_data.copy()
    ml["warranty_claim"] = np.random.RandomState(7).permutation(ml["warranty_claim"].values)
    tr = train_and_select(ml, random_state=3)
    auc = roc_auc_score(tr["y_test"], tr["best_proba"])
    assert 0.25 <= auc <= 0.75, f"AUC {auc:.2f} on shuffled labels suggests leakage"


def test_train_val_test_vehicles_never_overlap_in_training_result(learned_pipe):
    tr = learned_pipe.train_result
    train, val, test = set(tr["X_train"].index), set(tr["X_val"].index), set(tr["X_test"].index)
    assert not (train & val) and not (train & test) and not (val & test)


def test_label_generation_is_reproducible_with_a_seed():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"SNS_A": rng.normal(size=200), "SNS_B": rng.normal(size=200)})
    a, _ = generate_warranty_labels(df, random_state=5)
    b, _ = generate_warranty_labels(df, random_state=5)
    c, _ = generate_warranty_labels(df, random_state=6)
    assert a["warranty_claim"].equals(b["warranty_claim"])
    assert not a["warranty_claim"].equals(c["warranty_claim"])
