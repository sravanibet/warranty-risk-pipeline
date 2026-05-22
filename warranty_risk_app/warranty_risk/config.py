import json
from pathlib import Path

# Default: data/config relative to the project root (one level above warranty_risk_app/)
_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "data" / "config" / "plant_config.json"


def load_config(config_path: Path = None) -> dict:
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    with open(path) as f:
        return json.load(f)


def get_stations(config: dict) -> list:
    return config["stations"]
