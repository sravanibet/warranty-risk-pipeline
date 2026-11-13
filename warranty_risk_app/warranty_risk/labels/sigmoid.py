import numpy as np
import pandas as pd

DEFAULT_RISK_WEIGHT = 1.0


def _weights_from_config(config: dict) -> dict:
    # Build {sensor_uid: weight} from plant config. A sensor's "risk_weight" field
    # may be a number, "excluded" (weight 0), or absent (falls back to default).
    weights = {}
    for station in config.get("stations", []):
        for sensor in station.get("sensors", []):
            uid = sensor.get("sensor_uid")
            if not uid:
                continue
            raw = sensor.get("risk_weight", DEFAULT_RISK_WEIGHT)
            weights[uid] = 0.0 if raw == "excluded" else float(raw)
    return weights


def generate_warranty_labels(
    feature_df: pd.DataFrame,
    config: dict = None,
    k: float = 10.0,
    max_claim_rate: float = 0.12,
    random_state: int = None,
) -> pd.DataFrame:
    # Add a 'warranty_claim' column using a stochastic sigmoid approach.
    #
    # Sensor risk weights come from `config` (each sensor's optional "risk_weight"
    # field); sensors without one default to DEFAULT_RISK_WEIGHT. If no config is
    # given, every sensor column present in feature_df gets the default weight.
    #
    # Avoids the circular dependency of NN top-10%:
    #   p_claim = sigmoid(k × (risk_norm − threshold))
    #   warranty_claim ~ Binomial(1, p_claim)
    #
    # Threshold is scanned adaptively until E[p_claim] <= max_claim_rate.
    # Returns feature_df with the new column appended (does not mutate input).
    df = feature_df.copy()

    weights = _weights_from_config(config) if config else {}
    sensor_cols = [c for c in df.columns if c != "warranty_claim"]
    risk_seed = sum(
        df[col] * weights.get(col, DEFAULT_RISK_WEIGHT)
        for col in sensor_cols
        if weights.get(col, DEFAULT_RISK_WEIGHT) != 0.0
    )
    risk_norm = (risk_seed - risk_seed.min()) / (risk_seed.max() - risk_seed.min())

    chosen_t = 0.90
    for t in [round(v, 2) for v in list(np.arange(0.50, 0.99, 0.01))]:
        p_test = 1 / (1 + np.exp(-k * (risk_norm - t)))
        if p_test.mean() <= max_claim_rate:
            chosen_t = t
            break

    p_claim        = 1 / (1 + np.exp(-k * (risk_norm - chosen_t)))
    warranty_claim = np.random.default_rng(random_state).binomial(1, p_claim)
    df["warranty_claim"] = warranty_claim

    meta = {
        "sigmoid_threshold": chosen_t,
        "claim_count":       int(warranty_claim.sum()),
        "claim_rate":        float(warranty_claim.mean()),
        "p_claim_min":       float(p_claim.min()),
        "p_claim_max":       float(p_claim.max()),
    }
    return df, meta
