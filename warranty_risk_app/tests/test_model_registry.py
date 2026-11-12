import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from warranty_risk.agent import model_registry


def _fake_train_result():
    X = np.random.RandomState(0).normal(size=(40, 3))
    y = (X[:, 0] > 0).astype(int)
    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    return {
        "best_model": model, "scaler": scaler, "best_name": "RF",
        "feat_cols": ["f1", "f2", "f3"],
    }


def test_save_creates_expected_files(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "_MODELS_DIR", tmp_path)
    save_dir = model_registry.save(_fake_train_result(), f1=0.8)
    assert (save_dir / "model.joblib").exists()
    assert (save_dir / "scaler.joblib").exists()
    assert (save_dir / "metadata.json").exists()
    assert (tmp_path / "latest.json").exists()


def test_load_latest_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "_MODELS_DIR", tmp_path)
    tr = _fake_train_result()
    model_registry.save(tr, f1=0.75)

    record = model_registry.load_latest()
    assert record is not None
    assert record.best_name == "RF"
    assert record.f1 == 0.75
    assert record.feat_cols == ["f1", "f2", "f3"]
    # the loaded model should produce identical predictions to the original
    X_test = np.random.RandomState(1).normal(size=(5, 3))
    np.testing.assert_array_equal(
        record.model.predict(X_test), tr["best_model"].predict(X_test)
    )


def test_load_latest_returns_none_when_nothing_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "_MODELS_DIR", tmp_path)
    assert model_registry.load_latest() is None


def test_list_saved_orders_most_recent_first(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "_MODELS_DIR", tmp_path)
    model_registry.save(_fake_train_result(), f1=0.5)
    model_registry.save(_fake_train_result(), f1=0.9)
    records = model_registry.list_saved()
    assert len(records) == 2
    assert records[0]["saved_at"] >= records[1]["saved_at"]


def test_preprocessing_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "_MODELS_DIR", tmp_path)
    prep = {"category_map": {"dent": 0, "none": 1}, "sensor_uids": ["A", "B"], "max_seq_len": 3}
    model_registry.save(_fake_train_result(), f1=0.7, preprocessing=prep)
    assert model_registry.load_latest().preprocessing == prep


def test_old_save_without_preprocessing_still_loads(tmp_path, monkeypatch):
    monkeypatch.setattr(model_registry, "_MODELS_DIR", tmp_path)
    model_registry.save(_fake_train_result(), f1=0.7)
    assert model_registry.load_latest().preprocessing is None
