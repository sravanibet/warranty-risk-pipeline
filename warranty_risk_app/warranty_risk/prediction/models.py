import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve,
)

SCALE_NEEDED = {"MLP", "LR", "SVM", "KNN"}

_CANDIDATES = {
    "RF":   RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    "GB":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "LR":   LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "MLP":  MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "DT":   DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "Ada":  AdaBoostClassifier(n_estimators=100, random_state=42),
    "KNN":  KNeighborsClassifier(n_neighbors=5),
    "NB":   GaussianNB(),
    "SVM":  SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42),
}


def _optimal_threshold(proba: np.ndarray, y_true: np.ndarray) -> float:
    # F1-maximising threshold. Call on VALIDATION data, never on the test set.
    if len(np.unique(y_true)) < 2:
        return 0.5
    pr, rc, th = precision_recall_curve(y_true, proba)
    f1s = 2 * pr * rc / (pr + rc + 1e-9)
    return float(th[np.argmax(f1s[:-1])]) if len(th) > 0 else 0.5


def _safe_auc(y_true, proba) -> float:
    return float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan")


def split_vehicles(y: pd.Series, val_size: float = 0.2, test_size: float = 0.2,
                   random_state: int = 42) -> dict:
    # Split vehicles (the index of y) into disjoint train / validation / test sets,
    # stratified on the label.
    #
    # Tolerance optimisation and model selection use only the train and validation
    # sets; the test set stays untouched and is used only for the final metrics.
    # Do this BEFORE anything that looks at labels (tolerance optimisation,
    # feature selection, imputation) so those can be fit on `train` only.
    #
    # Returns {"train": Index, "val": Index, "test": Index}.
    # Split by vehicle BEFORE anything uses labels, so the tolerance optimiser
    # (which uses claim labels) can be fit on train vehicles only and test
    # vehicles stay completely unseen.
    # We split on positional indices because the vehicle ids can be an
    # Arrow-backed pandas Index, which train_test_split can't index directly.
    idx = y.index
    pos = np.arange(len(y))
    labels = y.to_numpy()
    tv_pos, test_pos = train_test_split(
        pos, test_size=test_size, stratify=labels, random_state=random_state
    )
    rel_val = val_size / (1.0 - test_size)
    train_pos, val_pos = train_test_split(
        tv_pos, test_size=rel_val, stratify=labels[tv_pos], random_state=random_state
    )
    train_idx, val_idx, test_idx = idx[train_pos], idx[val_pos], idx[test_pos]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def train_and_select(
    ml_data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    split: dict = None,
    val_size: float = 0.2,
) -> dict:
    # Train all candidate classifiers on the train vehicles, choose the best model
    # AND its decision threshold on the validation vehicles, and report metrics
    # once on the untouched test vehicles.
    #
    # split: optional {"train","val","test"} vehicle indices from split_vehicles
    # (pass the one used to fit the tolerance optimiser so no test vehicle ever
    # influenced a fitted component). If omitted, a new split is made here.
    #
    # results_df columns: Model, Val_F1 (selection criterion), then F1 / Precision /
    # Recall / AUC / Threshold, which are TEST-set metrics at the validation-chosen
    # threshold. Sorted by Val_F1.
    #
    # Also returns best_model, best_name, best_threshold, best_proba, best_preds
    # (test set), X/y for train, val and test (plus scaled arrays), scaler,
    # feat_cols, tol_cols and the split used.
    tol_cols  = ["within_tolerance_all", "out_of_tolerance_count"]
    feat_cols = [c for c in ml_data.columns if c != "warranty_claim"]

    y_all = ml_data["warranty_claim"].astype(int)
    if split is None:
        split = split_vehicles(y_all, val_size=val_size, test_size=test_size,
                               random_state=random_state)

    X_train, y_train = ml_data.loc[split["train"], feat_cols], y_all.loc[split["train"]]
    X_val,   y_val   = ml_data.loc[split["val"],   feat_cols], y_all.loc[split["val"]]
    X_test,  y_test  = ml_data.loc[split["test"],  feat_cols], y_all.loc[split["test"]]

    # The scaler is fit on train only; val and test are just transformed with
    # it, so their statistics never leak into training.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    rows = []
    for name, base_clf in _CANDIDATES.items():
        # clone() gives each run a fresh, unfitted model so nothing is
        # carried over between calls or between splits.
        clf = clone(base_clf)
        scaled = name in SCALE_NEEDED
        Xtr = X_train_scaled if scaled else X_train.values
        Xva = X_val_scaled   if scaled else X_val.values
        Xte = X_test_scaled  if scaled else X_test.values
        clf.fit(Xtr, y_train)

        proba_val = clf.predict_proba(Xva)[:, 1]
        # The decision threshold is chosen on the validation set. Choosing it
        # on test would make the test score look better than it really is.
        thr = _optimal_threshold(proba_val, y_val)                  # threshold: validation
        val_f1 = f1_score(y_val, (proba_val >= thr).astype(int), zero_division=0)

        proba = clf.predict_proba(Xte)[:, 1]                        # reported: test
        preds = (proba >= thr).astype(int)
        rows.append({
            "Model":     name,
            "Val_F1":    round(val_f1, 3),
            "F1":        round(f1_score(y_test, preds, zero_division=0), 3),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "Recall":    round(recall_score(y_test, preds, zero_division=0), 3),
            "AUC":       round(_safe_auc(y_test, proba), 3),
            "Threshold": round(thr, 3),
            "_clf":      clf,
        })

    # The best model is picked by validation F1. The test set is only used to
    # report the final numbers, once.
    results_df = pd.DataFrame(rows).sort_values("Val_F1", ascending=False).reset_index(drop=True)

    best_row       = results_df.iloc[0]
    best_name      = best_row["Model"]
    best_model     = best_row["_clf"]
    best_threshold = best_row["Threshold"]

    best_Xte   = X_test_scaled if best_name in SCALE_NEEDED else X_test.values
    best_proba = best_model.predict_proba(best_Xte)[:, 1]
    best_preds = (best_proba >= best_threshold).astype(int)

    return {
        "best_model":      best_model,
        "best_name":       best_name,
        "best_threshold":  best_threshold,
        "best_proba":      best_proba,
        "best_preds":      best_preds,
        "X_train":         X_train,
        "X_val":           X_val,
        "X_test":          X_test,
        "y_train":         y_train,
        "y_val":           y_val,
        "y_test":          y_test,
        "X_train_scaled":  X_train_scaled,
        "X_val_scaled":    X_val_scaled,
        "X_test_scaled":   X_test_scaled,
        "scaler":          scaler,
        "results_df":      results_df.drop(columns=["_clf"]),
        "feat_cols":       feat_cols,
        "tol_cols":        tol_cols,
        "split":           split,
    }


def compare_baseline_enhanced(train_result: dict) -> dict:
    # Retrain clone(best_model) twice (threshold chosen on validation, metrics on test):
    #   - Baseline: time-series features only (no tolerance columns)
    #   - Enhanced: time-series + tolerance features
    #
    # Returns metrics and predictions for both variants.
    best_model    = train_result["best_model"]
    best_name     = train_result["best_name"]
    X_train       = train_result["X_train"]
    X_val         = train_result["X_val"]
    X_test        = train_result["X_test"]
    y_train       = train_result["y_train"]
    y_val         = train_result["y_val"]
    y_test        = train_result["y_test"]
    X_tr_sc       = train_result["X_train_scaled"]
    X_va_sc       = train_result["X_val_scaled"]
    X_te_sc       = train_result["X_test_scaled"]
    tol_cols      = train_result["tol_cols"]
    feat_cols     = train_result["feat_cols"]
    needs_scale   = best_name in SCALE_NEEDED

    # Column indices for tolerance features inside feat_cols
    tol_idx = [i for i, c in enumerate(feat_cols) if c in tol_cols]
    ts_idx  = [i for i, c in enumerate(feat_cols) if c not in tol_cols]

    def _get_arrays(idx_list):
        if needs_scale:
            return X_tr_sc[:, idx_list], X_va_sc[:, idx_list], X_te_sc[:, idx_list]
        return (X_train.values[:, idx_list], X_val.values[:, idx_list],
                X_test.values[:, idx_list])

    results = {}
    for label, idx in [("Baseline", ts_idx), ("Enhanced", list(range(len(feat_cols))))]:
        Xtr, Xva, Xte = _get_arrays(idx)
        m = clone(best_model)
        m.fit(Xtr, y_train)
        thr   = _optimal_threshold(m.predict_proba(Xva)[:, 1], y_val)   # validation
        proba = m.predict_proba(Xte)[:, 1]                                # reported on test
        preds = (proba >= thr).astype(int)
        results[label] = {
            "model":    m,
            "proba":    proba,
            "preds":    preds,
            "threshold": thr,
            "f1":       round(f1_score(y_test, preds, zero_division=0), 3),
            "precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "recall":   round(recall_score(y_test, preds, zero_division=0), 3),
            "auc":      round(_safe_auc(y_test, proba), 3),
        }

    return {"comparison": results, "y_test": y_test, "best_name": best_name}
