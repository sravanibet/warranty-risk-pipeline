"""
GA Tolerance Optimizer — core code copied verbatim from
ToleranceOptimization_June13_2025_ver2.py; only the I/O wrapper differs.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize


class ToleranceOpt(Problem):
    def __init__(self, data, sensors, claim_types, C_FA, C_FN, xl, xu):
        self.data          = data
        self.sensors       = sensors
        self.claim_types   = [c for c in claim_types if c != 0]
        self.C_FA          = C_FA
        self.C_FN          = C_FN
        self.non_claim_arr = data[data["Claim Type"] == 0][sensors].values
        self.claim_arrs    = {c: data[data["Claim Type"] == c][sensors].values
                              for c in self.claim_types}
        super().__init__(n_var=2 * len(sensors), n_obj=1,
                         n_constr=len(sensors), xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X    = np.atleast_2d(X)
        costs, constraints = [], []
        n    = len(self.sensors)
        for xi in X:
            lowers_i = xi[:n]
            uppers_i = xi[n:]
            constraints.append(lowers_i - uppers_i)
            A  = (self.non_claim_arr < lowers_i) | (self.non_claim_arr > uppers_i)
            FA = np.any(A, axis=1).sum()
            total_cost = 0.0
            for c, arr in self.claim_arrs.items():
                B       = (arr >= lowers_i) & (arr <= uppers_i)
                FN      = np.all(B, axis=1).sum()
                FA_norm = FA / self.non_claim_arr.shape[0]
                FN_norm = FN / arr.shape[0] if arr.shape[0] > 0 else 0
                total_cost += FA_norm * self.C_FA[c] + FN_norm * self.C_FN[c]
            costs.append(total_cost)
        out["F"] = np.array(costs).reshape(-1, 1)
        out["G"] = np.array(constraints)


def run_ga_optimization(
    feature_df: pd.DataFrame,
    sensor_cols: list,
    sensor_rename: dict,
    sensor_rename_inv: dict,
    n_generations: int = 50,
    pop_size: int = 200,
    c_fa: float = 0.1,
    c_fn: float = 2.0,
    verbose: bool = False,
) -> dict:
    """
    Run the genetic algorithm on feature_df and return optimized bounds.

    Returns a dict with keys:
        lowers_opt, uppers_opt, df_results, stats,
        sensors_opt, C_FA, C_FN, success (bool)
    """
    # Build optimizer input (sensor_1..N + Claim Type)
    opt_df = feature_df[sensor_cols + ["warranty_claim"]].copy()
    opt_df = opt_df.rename(columns=sensor_rename)
    opt_df = opt_df.rename(columns={"warranty_claim": "Claim Type"})
    opt_df["Claim Type"] = opt_df["Claim Type"].astype(int)

    sensors_opt = [c for c in opt_df.columns if c.startswith("sensor_")]
    claim_types = sorted(opt_df["Claim Type"].unique())
    C_FA = {c: c_fa * c for c in claim_types if c != 0}
    C_FN = {c: c_fn * c for c in claim_types if c != 0}

    X_opt = opt_df[sensors_opt]
    y_opt = opt_df["Claim Type"]
    X_tr, _, y_tr, _ = train_test_split(X_opt, y_opt, stratify=y_opt,
                                         test_size=0.2, random_state=42)
    train_data = X_tr.copy()
    train_data["Claim Type"] = y_tr

    stats = {
        s: {"min": train_data[s].min(), "max": train_data[s].max(),
            "mean": train_data[s].mean(), "std": train_data[s].std()}
        for s in sensors_opt
    }
    original_limits = {
        s: (stats[s]["mean"] - 2 * stats[s]["std"],
            stats[s]["mean"] + 2 * stats[s]["std"])
        for s in sensors_opt
    }

    lower_xl = [stats[s]["min"] - 0.5 * stats[s]["std"] for s in sensors_opt]
    upper_xl = [stats[s]["max"] + 0.5 * stats[s]["std"] for s in sensors_opt]
    xl = np.array(lower_xl + lower_xl)
    xu = np.array(upper_xl + upper_xl)

    problem   = ToleranceOpt(train_data, sensors_opt, claim_types, C_FA, C_FN, xl, xu)
    algorithm = GA(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PolynomialMutation(prob=0.2, eta=20),
        eliminate_duplicates=True,
    )
    result = minimize(problem, algorithm, get_termination("n_gen", n_generations),
                      seed=42, verbose=verbose, n_proc=4)

    if result.X is None:
        return {"success": False}

    x_opt     = result.X[0] if result.X.ndim > 1 else result.X
    n         = len(sensors_opt)
    lowers_opt = x_opt[:n]
    uppers_opt = x_opt[n:]

    df_results = pd.DataFrame({
        "Sensor":     sensors_opt,
        "Real_Name":  [sensor_rename_inv[s] for s in sensors_opt],
        "Lower":       lowers_opt,
        "Upper":       uppers_opt,
        "Target":      (lowers_opt + uppers_opt) / 2,
        "Orig_Lower":  [original_limits[s][0] for s in sensors_opt],
        "Orig_Upper":  [original_limits[s][1] for s in sensors_opt],
    })

    return {
        "success":     True,
        "lowers_opt":  lowers_opt,
        "uppers_opt":  uppers_opt,
        "df_results":  df_results,
        "stats":       stats,
        "sensors_opt": sensors_opt,
        "C_FA":        C_FA,
        "C_FN":        C_FN,
    }
