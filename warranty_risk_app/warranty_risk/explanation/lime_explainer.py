import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


def _get_explainer(X_train: np.ndarray, feature_names: list) -> LimeTabularExplainer:
    return LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["No Claim", "Claim"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def explain_vehicle(
    vehicle_idx: int,
    X_test: np.ndarray,
    X_train: np.ndarray,
    feature_names: list,
    predict_fn,
    num_features: int = 10,
) -> dict:
    """
    Generate a LIME explanation for a single test vehicle.

    Parameters
    ----------
    vehicle_idx  : row index into X_test
    X_test       : scaled/raw test feature array
    X_train      : scaled/raw train feature array (for explainer background)
    feature_names: list of feature column names
    predict_fn   : model.predict_proba (callable)
    num_features : top features to show

    Returns
    -------
    dict with keys:
        lime_df     — DataFrame of feature contributions
        proba       — claim probability (float)
        exp         — raw LimeExplanation object
    """
    explainer = _get_explainer(X_train, feature_names)

    instance = X_test[vehicle_idx]
    exp  = explainer.explain_instance(instance, predict_fn, num_features=num_features)
    proba = predict_fn(instance.reshape(1, -1))[0, 1]

    lime_list = exp.as_list()
    lime_df = pd.DataFrame(lime_list, columns=["Feature", "Weight"])
    lime_df["Direction"] = lime_df["Weight"].apply(lambda w: "Increases risk" if w > 0 else "Decreases risk")
    lime_df = lime_df.sort_values("Weight", key=abs, ascending=False)

    return {"lime_df": lime_df, "proba": float(proba), "exp": exp}


def plot_lime_instance(lime_df: pd.DataFrame, proba: float, vehicle_id: str = "") -> plt.Figure:
    """
    Three-panel LIME bar chart:
      1. Combined (orange=positive, blue=negative)
      2. Positive contributors only
      3. Negative contributors only
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2), dpi=150)
    title = f"LIME Explanation — {vehicle_id}  (p_claim={proba:.3f})" if vehicle_id else f"LIME Explanation (p_claim={proba:.3f})"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    pos = lime_df[lime_df["Weight"] >= 0]
    neg = lime_df[lime_df["Weight"] <  0]

    # Panel 1: combined
    ax = axes[0]
    colors = ["#E67E22" if w >= 0 else "#2980B9" for w in lime_df["Weight"]]
    ax.barh(lime_df["Feature"], lime_df["Weight"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Combined Impact")
    ax.invert_yaxis()

    # Panel 2: positive
    ax = axes[1]
    ax.barh(pos["Feature"], pos["Weight"], color="#E67E22")
    ax.set_title("Risk Increasing Features")
    ax.invert_yaxis()

    # Panel 3: negative
    ax = axes[2]
    ax.barh(neg["Feature"], neg["Weight"].abs(), color="#2980B9")
    ax.set_title("Risk Decreasing Features")
    ax.invert_yaxis()

    for ax in axes:
        ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    return fig


def global_lime(
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    predict_fn,
    n_high: int = 5,
    n_low: int = 5,
    num_features: int = 10,
) -> dict:
    """
    Aggregate LIME over n_high high-risk + n_low low-risk test vehicles.

    Returns a dict with:
        global_df  — DataFrame of mean absolute LIME weights per feature
        fig        — bar chart of top features
    """
    probas = predict_fn(X_test)[:, 1]
    sorted_idx = np.argsort(probas)

    low_idx  = sorted_idx[:n_low]
    high_idx = sorted_idx[-n_high:]
    sample_idx = np.concatenate([high_idx, low_idx])

    explainer = _get_explainer(X_train, feature_names)
    all_weights = {}

    for idx in sample_idx:
        exp   = explainer.explain_instance(X_test[idx], predict_fn, num_features=num_features)
        for feat, w in exp.as_list():
            all_weights.setdefault(feat, []).append(w)

    agg = {feat: np.mean(ws) for feat, ws in all_weights.items()}
    global_df = (
        pd.DataFrame(list(agg.items()), columns=["Feature", "MeanWeight"])
        .sort_values("MeanWeight", key=abs, ascending=False)
        .head(num_features)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
    colors = ["#E67E22" if w >= 0 else "#2980B9" for w in global_df["MeanWeight"]]
    ax.barh(global_df["Feature"], global_df["MeanWeight"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Global LIME — Top {num_features} Features ({n_high} high-risk + {n_low} low-risk vehicles)")
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()

    return {"global_df": global_df, "fig": fig}
