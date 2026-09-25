"""
Closes the loop between predictions and outcomes: after each Learning run,
checks how well each sensor's out-of-tolerance signal actually correlated
with real warranty claims, and nudges that sensor's risk_weight accordingly.

This is the agent's self-improvement mechanism — each run on new data makes
its own risk weighting a little more accurate than the last, instead of
staying fixed forever. It only adjusts *scoring weights*, never equipment or
process parameters — a human still confirms and acts on recommendations.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_WEIGHT = 0.1
MAX_WEIGHT = 5.0


@dataclass
class WeightAdjustment:
    sensor_uid: str
    old_weight: float
    new_weight: float
    correlation: float


def compute_weight_adjustments(
    feature_df: pd.DataFrame,
    ga_result: dict,
    sensor_rename_inv: dict,
    config: dict,
    step: float = 0.15,
) -> list:
    """
    For each GA-optimized sensor, correlate "was this vehicle's reading out
    of the optimized tolerance band" against the actual warranty_claim label.
    A sensor whose out-of-tolerance flag tracks real claims more than its
    current weight assumes gets nudged up; one that doesn't predict claims
    at all gets nudged down. Returns a list of WeightAdjustment (empty if the
    GA didn't succeed or there's no label column to compare against).
    """
    if not ga_result.get("success") or "warranty_claim" not in feature_df.columns:
        return []

    y = feature_df["warranty_claim"].astype(float)
    if y.std() == 0:
        return []

    sensor_weights = _current_weights(config)
    excluded = _explicitly_excluded(config)

    adjustments = []
    for s, lo, hi in zip(ga_result["sensors_opt"], ga_result["lowers_opt"], ga_result["uppers_opt"]):
        real = sensor_rename_inv.get(s, s)
        if real not in feature_df.columns or real in excluded:
            continue

        out_of_tol = (~feature_df[real].between(lo, hi)).astype(float)
        if out_of_tol.std() == 0:
            continue

        corr = float(np.corrcoef(out_of_tol, y)[0, 1])
        if np.isnan(corr):
            continue

        old_weight = sensor_weights.get(real, 1.0)
        new_weight = float(np.clip(old_weight + step * corr, MIN_WEIGHT, MAX_WEIGHT))

        adjustments.append(WeightAdjustment(
            sensor_uid=real,
            old_weight=round(old_weight, 3),
            new_weight=round(new_weight, 3),
            correlation=round(corr, 3),
        ))

    return adjustments


def apply_weight_adjustments(config: dict, adjustments: list) -> dict:
    """Writes adjusted weights back into a copy of the config's sensors."""
    by_uid = {a.sensor_uid: a.new_weight for a in adjustments}
    for station in config.get("stations", []):
        for sensor in station.get("sensors", []):
            uid = sensor.get("sensor_uid")
            if uid in by_uid:
                sensor["risk_weight"] = by_uid[uid]
    return config


def _explicitly_excluded(config: dict) -> set:
    """Sensors marked risk_weight: "excluded" are a deliberate domain judgment
    (e.g. cosmetic, no functional correlation) — the feedback loop leaves them alone."""
    excluded = set()
    for station in config.get("stations", []):
        for sensor in station.get("sensors", []):
            if sensor.get("risk_weight") == "excluded":
                excluded.add(sensor.get("sensor_uid"))
    return excluded


def _current_weights(config: dict) -> dict:
    weights = {}
    for station in config.get("stations", []):
        for sensor in station.get("sensors", []):
            uid = sensor.get("sensor_uid")
            if not uid:
                continue
            raw = sensor.get("risk_weight", 1.0)
            weights[uid] = 0.0 if raw == "excluded" else float(raw)
    return weights
