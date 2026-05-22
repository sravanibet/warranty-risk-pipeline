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
    pr, rc, th = precision_recall_curve(y_true, proba)
    f1s = 2 * pr * rc / (pr + rc + 1e-9)
    return float(th[np.argmax(f1s[:-1])]) if len(th) > 0 else 0.5


def train_and_select(
    ml_data: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """
    Train all candidate classifiers on (ml_data minus labels), evaluate on
    hold-out split, and return the best model selected by F1 score.

    Returns a dict with keys:
        best_model, best_name, best_threshold,
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler,
        results_df          (all models sorted by F1)
    """
    tol_cols  = ["within_tolerance_all", "out_of_tolerance_count"]
    # Tolerance features are included in training — this is the tolerance-enhanced model
    feat_cols = [c for c in ml_data.columns if c != "warranty_claim"]

    X = ml_data[feat_cols]
    y = ml_data["warranty_claim"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    rows = []
    for name, clf in _CANDIDATES.items():
        Xtr = X_train_scaled if name in SCALE_NEEDED else X_train.values
        Xte = X_test_scaled  if name in SCALE_NEEDED else X_test.values
        clf.fit(Xtr, y_train)
        proba = clf.predict_proba(Xte)[:, 1]
        thr   = _optimal_threshold(proba, y_test)
        preds = (proba >= thr).astype(int)
        rows.append({
            "Model":     name,
            "F1":        round(f1_score(y_test, preds, zero_division=0), 3),
            "Precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "Recall":    round(recall_score(y_test, preds, zero_division=0), 3),
            "AUC":       round(roc_auc_score(y_test, proba), 3),
            "Threshold": round(thr, 3),
            "_clf":      clf,
        })

    results_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    best_row       = results_df.iloc[0]
    best_name      = best_row["Model"]
    best_model     = best_row["_clf"]
    best_threshold = best_row["Threshold"]

    # Pre-compute best model predictions for the app to use directly
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
        "X_test":          X_test,
        "y_train":         y_train,
        "y_test":          y_test,
        "X_train_scaled":  X_train_scaled,
        "X_test_scaled":   X_test_scaled,
        "scaler":          scaler,
        "results_df":      results_df.drop(columns=["_clf"]),
        "feat_cols":       feat_cols,
        "tol_cols":        tol_cols,
    }


def compare_baseline_enhanced(train_result: dict) -> dict:
    """
    Retrain clone(best_model) twice:
      - Baseline: time-series features only (no tolerance columns)
      - Enhanced: time-series + tolerance features

    Returns metrics and predictions for both variants.
    """
    best_model    = train_result["best_model"]
    best_name     = train_result["best_name"]
    X_train       = train_result["X_train"]
    X_test        = train_result["X_test"]
    y_train       = train_result["y_train"]
    y_test        = train_result["y_test"]
    X_tr_sc       = train_result["X_train_scaled"]
    X_te_sc       = train_result["X_test_scaled"]
    tol_cols      = train_result["tol_cols"]
    feat_cols     = train_result["feat_cols"]
    needs_scale   = best_name in SCALE_NEEDED

    # Column indices for tolerance features inside feat_cols
    tol_idx = [i for i, c in enumerate(feat_cols) if c in tol_cols]
    ts_idx  = [i for i, c in enumerate(feat_cols) if c not in tol_cols]

    def _get_arrays(idx_list):
        if needs_scale:
            return X_tr_sc[:, idx_list], X_te_sc[:, idx_list]
        return X_train.values[:, idx_list], X_test.values[:, idx_list]

    results = {}
    for label, idx in [("Baseline", ts_idx), ("Enhanced", list(range(len(feat_cols))))]:
        Xtr, Xte = _get_arrays(idx)
        m = clone(best_model)
        m.fit(Xtr, y_train)
        proba = m.predict_proba(Xte)[:, 1]
        thr   = _optimal_threshold(proba, y_test)
        preds = (proba >= thr).astype(int)
        results[label] = {
            "model":    m,
            "proba":    proba,
            "preds":    preds,
            "threshold": thr,
            "f1":       round(f1_score(y_test, preds, zero_division=0), 3),
            "precision": round(precision_score(y_test, preds, zero_division=0), 3),
            "recall":   round(recall_score(y_test, preds, zero_division=0), 3),
            "auc":      round(roc_auc_score(y_test, proba), 3),
        }

    return {"comparison": results, "y_test": y_test, "best_name": best_name}
