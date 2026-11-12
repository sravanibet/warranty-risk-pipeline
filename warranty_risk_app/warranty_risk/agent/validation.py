"""
Statistical validation for the Learning phase: reporting a single F1 from
one training run doesn't say anything about how stable that number is.

Each run repeats the WHOLE leakage-free procedure with a different seed:
new train / validation / test vehicle split, tolerance bounds re-fit on that
run's train vehicles only, model and threshold chosen on validation, metrics
reported on the held-out test vehicles. Reports mean +/- standard deviation
plus a trivial no-learning baseline, so a reported metric is defensible
rather than a single lucky (or unlucky) draw.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

_TOL_COLS = ["within_tolerance_all", "out_of_tolerance_count"]


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


def _baseline_f1(tr: dict) -> float:
    """
    Trivial baseline on the run's test vehicles: flag a vehicle as a claim if
    it is out of tolerance on at least one sensor (bounds fit on that run's
    train vehicles), with no learning. Without tolerance features, predicts
    no claims, which scores 0 F1 for the minority class.
    """
    X_test, y_test = tr["X_test"], tr["y_test"]
    if "out_of_tolerance_count" in X_test.columns:
        preds = (X_test["out_of_tolerance_count"] > 0).astype(int)
    else:
        preds = np.zeros(len(y_test), dtype=int)
    return float(f1_score(y_test, preds, zero_division=0))


def validate_learning(pipe, n_runs: int = 5, refit_ga: bool = True) -> ValidationResult:
    """
    Repeat the split / tolerance-fit / select / evaluate procedure n_runs times.

    refit_ga=True (default) re-optimises tolerance bounds on each run's own
    train vehicles, which is the honest estimate but costs one GA run per seed.
    refit_ga=False reuses the pipeline's existing bounds (fit on the pipeline's
    train split), which is faster but leaks labels from the other runs' test
    vehicles into the tolerance features; use it only for quick smoke checks.
    """
    from ..prediction.models import train_and_select, split_vehicles

    if pipe.ml_data is None or pipe.feature_df_pre_opt is None:
        raise RuntimeError("No trained data available — run Learning before validating.")

    base_df = pipe.feature_df_pre_opt                       # sensors + warranty_claim
    y = base_df["warranty_claim"].astype(int)
    ml_base = pipe.ml_data.drop(columns=_TOL_COLS, errors="ignore")

    per_run = []
    baseline_f1s = []
    for seed in range(n_runs):
        split = split_vehicles(y, random_state=seed)
        if refit_ga:
            _, fdf = pipe.fit_tolerance(base_df, split["train"])
            tol = [c for c in _TOL_COLS if c in fdf.columns]
            ml_data = ml_base.join(fdf[tol], how="left")
            ml_data[tol] = ml_data[tol].fillna(0)
        else:
            ml_data = pipe.ml_data

        tr = train_and_select(ml_data, split=split)
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
        baseline_f1s.append(_baseline_f1(tr))

    f1s = [r["F1"] for r in per_run]
    precisions = [r["Precision"] for r in per_run]
    recalls = [r["Recall"] for r in per_run]
    aucs = [r["AUC"] for r in per_run]

    return ValidationResult(
        n_runs=n_runs,
        f1_mean=float(np.mean(f1s)), f1_std=float(np.std(f1s)),
        precision_mean=float(np.mean(precisions)), precision_std=float(np.std(precisions)),
        recall_mean=float(np.mean(recalls)), recall_std=float(np.std(recalls)),
        auc_mean=float(np.nanmean(aucs)), auc_std=float(np.nanstd(aucs)),
        baseline_f1_mean=float(np.mean(baseline_f1s)),
        per_run=per_run,
    )
