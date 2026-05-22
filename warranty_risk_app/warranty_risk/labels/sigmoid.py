import numpy as np
import pandas as pd


# Sensor weights reflecting domain knowledge (paint thickness excluded —
# cosmetic variation has low correlation with functional warranty claims)
_WEIGHTS = {
    "SNS_VIBRATION_001":     2.0,
    "SNS_TORQUE_001":        1.5,
    "SNS_WELD_TEMP_001":     1.2,
    "SNS_VISUAL_DEFECT_001": 3.0,
    "SNS_CONTINUITY_001":    2.5,
}


def generate_warranty_labels(
    feature_df: pd.DataFrame,
    k: float = 10.0,
    max_claim_rate: float = 0.12,
) -> pd.DataFrame:
    """
    Add a 'warranty_claim' column using a stochastic sigmoid approach.

    Avoids the circular dependency of NN top-10%:
      p_claim = sigmoid(k × (risk_norm − threshold))
      warranty_claim ~ Binomial(1, p_claim)

    Threshold is scanned adaptively until E[p_claim] <= max_claim_rate.
    Returns feature_df with the new column appended (does not mutate input).
    """
    df = feature_df.copy()

    risk_seed = sum(
        df[col] * weight
        for col, weight in _WEIGHTS.items()
        if col in df.columns
    )
    risk_norm = (risk_seed - risk_seed.min()) / (risk_seed.max() - risk_seed.min())

    chosen_t = 0.90
    for t in [round(v, 2) for v in list(np.arange(0.50, 0.99, 0.01))]:
        p_test = 1 / (1 + np.exp(-k * (risk_norm - t)))
        if p_test.mean() <= max_claim_rate:
            chosen_t = t
            break

    p_claim        = 1 / (1 + np.exp(-k * (risk_norm - chosen_t)))
    warranty_claim = np.random.binomial(1, p_claim)
    df["warranty_claim"] = warranty_claim

    meta = {
        "sigmoid_threshold": chosen_t,
        "claim_count":       int(warranty_claim.sum()),
        "claim_rate":        float(warranty_claim.mean()),
        "p_claim_min":       float(p_claim.min()),
        "p_claim_max":       float(p_claim.max()),
    }
    return df, meta
