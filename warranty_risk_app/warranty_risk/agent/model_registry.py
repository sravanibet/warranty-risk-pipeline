"""
Persists a trained model between runs, so Learning doesn't have to retrain
from scratch every time the process restarts — and keeps every past version
instead of overwriting, so a worse model can be rolled back to a better one.

Each save writes a timestamped directory under outputs/models/ containing:
    model.joblib      — the trained sklearn estimator
    scaler.joblib      — the fitted StandardScaler
    metadata.json       — feat_cols, best_name, metrics, timestamp, and the
                          preprocessing needed to score new data (category map,
                          sensor layout) so a reloaded model encodes inputs
                          exactly as it was trained to

outputs/models/latest.json points at the most recent save's directory name,
so load_latest() doesn't need to list and sort directories itself.
"""
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "models"


@dataclass
class ModelRecord:
    model: object
    scaler: object
    feat_cols: list
    best_name: str
    f1: float
    saved_at: str
    path: Path
    preprocessing: dict = None   # category_map, sensor_uids, max_seq_len (None for old saves)


def save(train_result: dict, f1: float, preprocessing: dict = None) -> Path:
    """
    Saves the best trained model from a Learning run. Returns the save directory.
    preprocessing: JSON-serialisable dict with category_map, sensor_uids and
    max_seq_len from training time.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    save_dir = _MODELS_DIR / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(train_result["best_model"], save_dir / "model.joblib")
    joblib.dump(train_result["scaler"], save_dir / "scaler.joblib")

    metadata = {
        "best_name": train_result["best_name"],
        "feat_cols": list(train_result["feat_cols"]),
        "f1": f1,
        "saved_at": timestamp,
        "preprocessing": preprocessing,
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_MODELS_DIR / "latest.json", "w") as f:
        json.dump({"latest": timestamp}, f)

    return save_dir


def load_latest() -> ModelRecord:
    """Loads the most recently saved model. Returns None if nothing has been saved yet."""
    pointer_path = _MODELS_DIR / "latest.json"
    if not pointer_path.exists():
        return None
    with open(pointer_path) as f:
        timestamp = json.load(f)["latest"]
    return load(timestamp)


def load(timestamp: str) -> ModelRecord:
    """Loads a specific saved model by its timestamp directory name."""
    save_dir = _MODELS_DIR / timestamp
    with open(save_dir / "metadata.json") as f:
        metadata = json.load(f)

    return ModelRecord(
        model=joblib.load(save_dir / "model.joblib"),
        scaler=joblib.load(save_dir / "scaler.joblib"),
        feat_cols=metadata["feat_cols"],
        best_name=metadata["best_name"],
        f1=metadata["f1"],
        saved_at=metadata["saved_at"],
        path=save_dir,
        preprocessing=metadata.get("preprocessing"),
    )


def list_saved() -> list:
    """Returns metadata for every saved model, most recent first."""
    if not _MODELS_DIR.exists():
        return []
    records = []
    for d in _MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                records.append(json.load(f))
    return sorted(records, key=lambda r: r["saved_at"], reverse=True)
