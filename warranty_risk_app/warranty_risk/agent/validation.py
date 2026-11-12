"""
Statistical validation for the Learning phase: reporting a single F1 from
one training run doesn't say anything about how stable that number is.
This runs Learning N times with different random seeds and reports
mean +/- standard deviation, plus a trivial baseline for comparison, so a
reported metric is defensible rather than a single lucky (or unlucky) draw.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    n_runs: int
    f1_mean: float
    f1_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    auc_mean: float
    auc_std: float
    baseline_f1_mean: float
    per_run: list = field(default_factory=list)

    @property
    def lift_over_baseline(self) -> float:
        return self.f1_mean - self.baseline_f1_mean


def _simple_baseline_f1(ml_data: pd.DataFrame, feat_cols: list, test_size: float, random_state: int) -> float:
    """
    A trivial baseline to compare the trained ensemble against: flag a
    vehicle as a predicted claim if it's out of tolerance on at least one
    sensor (the `out_of_tolerance_count` column), with no learning at all.
    If that column isn't present, falls back to always predicting the
    majority class, which should score near 0 F1 for the minority class.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    y = ml_data["warranty_claim"].astype(int)
    X = ml_data[feat_cols]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if "out_of_tolerance_count" in X_test.columns:
        preds = (X_test["out_of_tolerance_count"] > 0).astype(int)
    else:
        preds = np.zeros(len(y_test), dtype=int)

    return float(f1_score(y_test, preds, zero_division=0))


def validate_learning(pipe, n_runs: int = 5, test_size: float = 0.25) -> ValidationResult:
    """
    Re-runs model training n_runs times on the pipeline's current ml_data
    with different random seeds (tolerance optimization and data are reused
    as-is — only the train/test split and model fitting vary), and reports
    mean +/- std for each metric, plus a baseline comparison.
    """
    from ..prediction.models import train_and_select

    if pipe.ml_data is None:
        raise RuntimeError("No trained data available — run Learning before validating.")

    per_run = []
    baseline_f1s = []
    for seed in range(n_runs):
        tr = train_and_select(pipe.ml_data, test_size=test_size, random_state=seed)
        results_df = tr["results_df"]
        best_row = results_df.loc[results_df["Model"] == tr["best_name"]].iloc[0]
        per_run.append({
            "seed": seed,
            "best_name": tr["best_name"],
            "F1": float(best_row["F1"]),
            "Precision": float(best_row["Precision"]),
            "Recall": float(best_row["Recall"]),
            "AUC": float(best_row["AUC"]),
        })
        baseline_f1s.append(
            _simple_baseline_f1(pipe.ml_data, tr["feat_cols"], test_size, seed)
        )

    f1s = [r["F1"] for r in per_run]
    precisions = [r["Precision"] for r in per_run]
    recalls = [r["Recall"] for r in per_run]
    aucs = [r["AUC"] for r in per_run]

    return ValidationResult(
        n_runs=n_runs,
        f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
        precision_mean=float(np.mean(precisions)), precision_std=float(np.std(precisions)),
        recall_mean=float(np.mean(recalls)), recall_std=float(np.std(recalls)),
        auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
        baseline_f1_mean=float(np.mean(baseline_f1s)),
        per_run=per_run,
    )
