"""
Warranty Risk Prototype — Streamlit UI
Run: streamlit run warranty_risk_app/app.py

Developed by: Noor-E Jannat, PhD Candidate
              Industrial & Systems Engineering (ISE) Department
              Wayne State University
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc as sk_auc

plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150

from warranty_risk.pipeline import WarrantyRiskPipeline
from warranty_risk.explanation.lime_explainer import plot_lime_instance, explain_vehicle, global_lime
from warranty_risk.config import load_config
from warranty_risk.optimization.ga_optimizer import run_ga_optimization
from warranty_risk.optimization.tolerance_features import compute_tolerance_features
from warranty_risk.prediction.models import train_and_select, SCALE_NEEDED
from warranty_risk.preprocessing.preprocessor import auto_preprocess

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Warranty Risk Pipeline",
    page_icon="🏭",
    layout="wide",
)
st.title("Warranty Risk Prediction Pipeline")

# ── Session state init — must come before sidebar ─────────────────────────────
for key, default in [
    ("pipe", None),
    ("sim_preprocess_report", None),
    ("ext", {}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Shared rendering helpers ──────────────────────────────────────────────────

def _render_preprocess_report(report: list, df_before, df_after):
    """Render the auto-preprocessing report with before/after comparison."""
    col_before, col_after = st.columns(2)
    col_before.metric("Rows before", f"{len(df_before):,}")
    col_after.metric("Rows after",   f"{len(df_after):,}")
    col_before.metric("Columns before", df_before.shape[1])
    col_after.metric("Columns after",   df_after.shape[1])

    st.divider()
    st.markdown("#### Step-by-Step Report")

    for item in report:
        status = item["status"]
        step   = item["step"]
        detail = item["detail"]
        if status == "ok":
            st.success(f"**{step}** — {detail}")
        elif status == "warning":
            st.warning(f"**{step}** — {detail}")
        else:
            st.info(f"**{step}** — {detail}")

    st.divider()
    st.markdown("#### Cleaned Data Preview (first 10 rows)")
    st.dataframe(df_after.head(10).round(4), use_container_width=True)

    st.markdown("#### Column Summary")
    null_counts = df_after.isnull().sum()
    summary_rows = []
    for col in df_after.columns:
        summary_rows.append({
            "Column":   col,
            "Dtype":    str(df_after[col].dtype),
            "Non-null": int(df_after[col].notna().sum()),
            "Nulls":    int(null_counts[col]),
            "Unique":   int(df_after[col].nunique()),
            "Mean":     round(float(df_after[col].mean()), 4) if pd.api.types.is_numeric_dtype(df_after[col]) else "—",
            "Std":      round(float(df_after[col].std()),  4) if pd.api.types.is_numeric_dtype(df_after[col]) else "—",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


def _render_ga_tab(ga_result: dict, feature_df: pd.DataFrame, c_fa: float):
    """Render Tolerance Optimization tab content."""
    st.subheader("GA Tolerance Optimization")
    if not ga_result.get("success"):
        st.error("GA optimization did not find a valid solution.")
        return

    df_res      = ga_result["df_results"]
    stats_ga    = ga_result["stats"]
    sensors_opt = ga_result["sensors_opt"]

    within_count = int(feature_df["within_tolerance_all"].sum())
    out_count    = len(feature_df) - within_count

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sensors optimized",          len(sensors_opt))
    col2.metric("Within all tolerances",      within_count)
    col3.metric(">=1 sensor out of tolerance", out_count)
    col4.metric("False-reject cost (C_FA)",   c_fa)

    st.markdown("**Optimized tolerance limits**")
    st.dataframe(df_res.round(3), use_container_width=True)

    # Plot 1: Raw tolerance limits
    st.markdown("**Original vs Optimized Tolerance Limits (raw values)**")
    fig_raw, ax_raw = plt.subplots(figsize=(8.0, 3.2))
    x = np.arange(len(df_res))
    w = 0.2
    ax_raw.bar(x - 1.5*w, df_res["Orig_Lower"], w, label="Orig Lower", color="#AED6F1")
    ax_raw.bar(x - 0.5*w, df_res["Orig_Upper"], w, label="Orig Upper", color="#85C1E9")
    ax_raw.bar(x + 0.5*w, df_res["Lower"],      w, label="Opt Lower",  color="#A9DFBF")
    ax_raw.bar(x + 1.5*w, df_res["Upper"],      w, label="Opt Upper",  color="#52BE80")
    ax_raw.set_xticks(x)
    ax_raw.set_xticklabels(
        [rn.replace("SNS_", "") for rn in df_res["Real_Name"]],
        rotation=45, ha="right", fontsize=8,
    )
    ax_raw.legend(fontsize=7)
    ax_raw.set_title("Original vs Optimized Tolerance Limits")
    fig_raw.tight_layout()
    st.pyplot(fig_raw)
    plt.close(fig_raw)

    # Plot 2: Z-score ranges
    st.markdown("**Normalized (Z-Score) Tolerance Ranges**")
    st.caption(
        "Z-score = (limit − sensor mean) / sensor std. "
        "Shows how many standard deviations each bound is from the mean."
    )
    short_labels  = [rn.replace("SNS_", "") for rn in df_res["Real_Name"]]
    orig_lower_z  = [(l - stats_ga[s]["mean"]) / stats_ga[s]["std"]
                     for s, l in zip(sensors_opt, df_res["Orig_Lower"])]
    orig_upper_z  = [(u - stats_ga[s]["mean"]) / stats_ga[s]["std"]
                     for s, u in zip(sensors_opt, df_res["Orig_Upper"])]
    opt_lower_z   = [(l - stats_ga[s]["mean"]) / stats_ga[s]["std"]
                     for s, l in zip(sensors_opt, df_res["Lower"])]
    opt_upper_z   = [(u - stats_ga[s]["mean"]) / stats_ga[s]["std"]
                     for s, u in zip(sensors_opt, df_res["Upper"])]

    x_z = np.arange(len(sensors_opt))
    fig_z, ax_z = plt.subplots(figsize=(8.0, 4.0))
    ax_z.vlines(x_z, orig_lower_z, orig_upper_z, colors="#85C1E9",
                linewidth=9, label="Original Range", alpha=0.65)
    ax_z.vlines(x_z + 0.12, opt_lower_z, opt_upper_z, colors="#52BE80",
                linewidth=9, label="Optimized Range", alpha=0.9)
    ax_z.scatter(x_z,        orig_lower_z, color="#85C1E9", marker="o", s=80, zorder=5)
    ax_z.scatter(x_z,        orig_upper_z, color="#85C1E9", marker="o", s=80, zorder=5)
    ax_z.scatter(x_z + 0.12, opt_lower_z,  color="#52BE80", marker="s", s=80, zorder=5)
    ax_z.scatter(x_z + 0.12, opt_upper_z,  color="#52BE80", marker="s", s=80, zorder=5)
    ax_z.axhline(0, linestyle="--", color="black", linewidth=0.8, alpha=0.5,
                 label="Sensor mean (Z=0)")
    ax_z.set_title("Normalized (Z-Score) Tolerance Ranges per Sensor",
                   fontweight="bold", fontsize=12)
    ax_z.set_xlabel("Sensor")
    ax_z.set_ylabel("Z-Score  (standard deviations from mean)")
    ax_z.set_xticks(x_z + 0.06)
    ax_z.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax_z.legend(fontsize=9)
    ax_z.grid(axis="y", linestyle="--", alpha=0.4)
    fig_z.tight_layout()
    st.pyplot(fig_z)
    plt.close(fig_z)

    # Tolerance status summary
    st.markdown("**Tolerance feature distribution**")
    if "within_tolerance_all" in feature_df.columns:
        total = len(feature_df)
        tol_summary = pd.DataFrame({
            "Status": ["Within all tolerances", ">=1 sensor out of tolerance"],
            "Vehicles": [within_count, out_count],
            "Percentage": [f"{within_count / total:.1%}", f"{out_count / total:.1%}"],
        })
        st.dataframe(tol_summary, use_container_width=True, hide_index=True)

        oot_series = (
            feature_df["out_of_tolerance_count"]
            .value_counts().sort_index().reset_index()
        )
        oot_series.columns = ["Sensors out of tolerance", "Vehicles"]
        st.markdown("**Out-of-tolerance sensor count per vehicle**")
        st.dataframe(oot_series, use_container_width=True, hide_index=True)


def _render_prediction_tab(train_result: dict, results_df: pd.DataFrame):
    """Render Model Training & Selection tab content."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    st.subheader("Model Training & Selection (Tolerance-Enhanced Features)")
    tr = train_result
    st.markdown(
        f"**Best model:** `{tr['best_name']}` (selected by F1) · "
        f"threshold: `{tr['best_threshold']:.3f}`"
    )
    st.dataframe(
        results_df.style.highlight_max(subset=["F1"], color="#A9DFBF"),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Best Model Performance")

    y_test     = tr["y_test"]
    best_proba = tr["best_proba"]
    best_preds = tr["best_preds"]

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("F1",        f"{f1_score(y_test, best_preds, zero_division=0):.3f}")
    mc2.metric("Precision", f"{precision_score(y_test, best_preds, zero_division=0):.3f}")
    mc3.metric("Recall",    f"{recall_score(y_test, best_preds, zero_division=0):.3f}")
    fpr_full, tpr_full, _ = roc_curve(y_test, best_proba)
    auc_val = sk_auc(fpr_full, tpr_full)
    mc4.metric("AUC", f"{auc_val:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.suptitle(f"{tr['best_name']} — Tolerance-Enhanced Model", fontsize=11)

    cm  = confusion_matrix(y_test, best_preds)
    ax  = axes[0]
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Claim", "Claim"])
    ax.set_yticklabels(["No Claim", "Claim"])
    cm_max = cm.max() if cm.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm_max / 2 else "black")

    ax = axes[1]
    ax.plot(fpr_full, tpr_full, lw=2, label=f"AUC = {auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_lime_tab(lime_result: dict, global_lime_result: dict, train_result: dict):
    """Render LIME Explanation tab content."""
    st.subheader("LIME Explanation")

    lr         = lime_result
    glr        = global_lime_result
    tr         = train_result
    lime_df    = lr["lime_df"]
    vehicle_id = lr.get("vehicle_id", "Unknown")

    st.markdown(
        f"### Vehicle: `{vehicle_id}`  —  "
        f"Predicted Warranty Claim Probability: **`{lr['proba']:.3f}` "
        f"({lr['proba'] * 100:.1f}%)**"
    )
    st.caption(
        "This is the highest-risk vehicle in the test set. "
        "LIME fits a local linear model around this vehicle to explain why the "
        "classifier assigned this risk score."
    )

    st.markdown("#### Feature Impact — Combined View")
    fig_comb, ax_c = plt.subplots(figsize=(8.0, max(3.2, len(lime_df) * 0.36)))
    colors_comb = ["#E67E22" if w >= 0 else "#2980B9" for w in lime_df["Weight"]]
    ax_c.barh(lime_df["Feature"], lime_df["Weight"], color=colors_comb)
    ax_c.axvline(0, color="black", linewidth=0.8)
    ax_c.set_title(
        f"Vehicle {vehicle_id} — p(claim) = {lr['proba']:.3f}",
        fontweight="bold", fontsize=11,
    )
    ax_c.set_xlabel("LIME Weight  (impact on prediction)")
    ax_c.invert_yaxis()
    ax_c.tick_params(axis="y", labelsize=8)
    fig_comb.tight_layout()
    st.pyplot(fig_comb)
    plt.close(fig_comb)

    st.caption(
        "**Orange** → pushes prediction toward Warranty Claim  |  "
        "**Blue** → pushes prediction away from Warranty Claim"
    )

    pos_df = lime_df[lime_df["Weight"] >= 0].copy()
    neg_df = lime_df[lime_df["Weight"] <  0].copy()
    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("#### Risk-Increasing Features")
        if pos_df.empty:
            st.info("No features increase risk for this vehicle.")
        else:
            fig_pos, ax_p = plt.subplots(figsize=(4.8, max(2.4, len(pos_df) * 0.44)))
            ax_p.barh(pos_df["Feature"], pos_df["Weight"], color="#E67E22")
            ax_p.set_title(f"Features Increasing Claim Risk\nVehicle {vehicle_id}", fontsize=10)
            ax_p.set_xlabel("LIME Weight  (-> higher risk)")
            ax_p.invert_yaxis()
            ax_p.tick_params(axis="y", labelsize=7)
            fig_pos.tight_layout()
            st.pyplot(fig_pos)
            plt.close(fig_pos)

    with col_neg:
        st.markdown("#### Risk-Decreasing Features")
        if neg_df.empty:
            st.info("No features decrease risk for this vehicle.")
        else:
            fig_neg, ax_n = plt.subplots(figsize=(4.8, max(2.4, len(neg_df) * 0.44)))
            ax_n.barh(neg_df["Feature"], neg_df["Weight"].abs(), color="#2980B9")
            ax_n.set_title(f"Features Reducing Claim Risk\nVehicle {vehicle_id}", fontsize=10)
            ax_n.set_xlabel("LIME Weight magnitude  (-> lower risk)")
            ax_n.invert_yaxis()
            ax_n.tick_params(axis="y", labelsize=7)
            fig_neg.tight_layout()
            st.pyplot(fig_neg)
            plt.close(fig_neg)

    st.markdown("**LIME feature contributions (tabular)**")
    st.dataframe(lr["lime_df"].round(4), use_container_width=True)

    st.divider()
    st.subheader("Global LIME — Top Features (5 high-risk + 5 low-risk vehicles)")
    st.caption(
        "LIME explanations aggregated over 5 highest-risk and 5 lowest-risk test vehicles."
    )

    test_ids   = list(tr["X_test"].index)
    best_proba = tr["best_proba"]
    sorted_idx = np.argsort(best_proba)
    high_idx   = sorted_idx[-5:][::-1]
    low_idx    = sorted_idx[:5]

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**High-risk vehicles (top 5)**")
        st.dataframe(
            pd.DataFrame([
                {"Serial Number": test_ids[i], "Risk Probability": f"{best_proba[i]:.4f}"}
                for i in high_idx
            ]),
            use_container_width=True, hide_index=True,
        )
    with g2:
        st.markdown("**Low-risk vehicles (bottom 5)**")
        st.dataframe(
            pd.DataFrame([
                {"Serial Number": test_ids[i], "Risk Probability": f"{best_proba[i]:.4f}"}
                for i in low_idx
            ]),
            use_container_width=True, hide_index=True,
        )

    st.markdown("#### Aggregated Feature Importance")
    st.pyplot(glr["fig"])
    plt.close(glr["fig"])

    st.markdown("**Global LIME weights (mean across sampled vehicles)**")
    st.dataframe(glr["global_df"].round(4), use_container_width=True)


# ── File-upload callbacks (run before next render, so sidebar sees updated state)
def _on_file_upload():
    f = st.session_state.get("_data_file")
    if f is None:
        st.session_state.ext.update({"raw_df": None, "label_col": None})
        return
    try:
        raw_df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        st.session_state.ext["raw_df"] = raw_df
        # Auto-detect a sensible default label column
        hints = ["warranty_claim", "claim", "warranty", "label", "target", "failure", "defect"]
        default_col = raw_df.columns[0]
        for hint in hints:
            matches = [c for c in raw_df.columns if hint.lower() in c.lower()]
            if matches:
                default_col = matches[0]
                break
        st.session_state.ext["label_col"] = default_col
    except Exception:
        st.session_state.ext.update({"raw_df": None, "label_col": None})


def _on_label_select():
    st.session_state.ext["label_col"] = st.session_state.get("_label_col")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Pipeline Settings")

    data_mode = st.radio(
        "Data Source",
        ["Simulate", "Upload External Data"],
        help="Simulate: generate synthetic manufacturing data from plant config. "
             "Upload: bring your own dataset.",
    )

    st.divider()

    if data_mode == "Simulate":
        n_vehicles = st.slider("Number of vehicles", 100, 1000, 500, step=50)

    n_generations = st.slider("GA generations",      10,  100,  50, step=10)
    pop_size      = st.slider("GA population size",  50,  500, 200, step=50)
    c_fa = st.number_input("Cost false-accept (C_FA)", value=0.1, step=0.05)
    c_fn = st.number_input("Cost false-reject (C_FN)", value=2.0, step=0.5)

    st.divider()

    if data_mode == "Simulate":
        run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)
    else:
        ext = st.session_state.ext
        can_run = (
            ext.get("raw_df") is not None
            and ext.get("label_col") is not None
        )
        run_btn = st.button(
            "Run Pipeline",
            type="primary",
            use_container_width=True,
            disabled=not can_run,
            help=None if can_run else "Upload a file and select the label column first.",
        )


# ── Run button logic ──────────────────────────────────────────────────────────
if run_btn:
    if data_mode == "Simulate":
        with st.spinner("Running simulation pipeline — this may take 1-2 minutes..."):
            pipe = WarrantyRiskPipeline(
                n_vehicles=n_vehicles,
                n_generations=n_generations,
                pop_size=pop_size,
                c_fa=c_fa,
                c_fn=c_fn,
            )
            pipe.run()

            # Run preprocessing on the labeled feature matrix for the report tab.
            # Simulated data is clean by design; the report will reflect that.
            _, sim_report = auto_preprocess(
                pipe.feature_df_pre_opt.copy(),
                label_col="warranty_claim",
            )
            st.session_state.pipe = pipe
            st.session_state.sim_preprocess_report = sim_report
        st.success("Simulation pipeline complete!")

    else:
        st.info(
            "The full analysis pipeline for uploaded data is currently under development. "
            "Please use the **Simulate** mode to explore all pipeline capabilities. "
            "You can still upload your file to review the data summary in the Data Ingestion tab."
        )


# ── Tabs — Simulate mode ──────────────────────────────────────────────────────
if data_mode == "Simulate":
    pipe = st.session_state.pipe
    tabs = st.tabs([
        "1 · Plant Setup & Sensors",
        "2 · Simulation",
        "3 · Labels",
        "4 · Data Preprocessing",
        "5 · Tolerance Optimization",
        "6 · Prediction",
        "7 · LIME Explanation",
    ])

    # ─── TAB 1: Plant Setup & Sensors ─────────────────────────────────────────
    with tabs[0]:
        st.subheader("Plant Configuration & Sensor Setup")
        config   = pipe.config if pipe else load_config()
        stations = config.get("stations", [])

        all_sensors  = [s for stn in stations for s in stn.get("sensors", [])]
        sensor_types = [s.get("data_type", "") for s in all_sensors]
        type_counts  = pd.Series(sensor_types).value_counts()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Plant",         config.get("plant_name", "—"))
        col2.metric("Plant ID",      config.get("plant_id", "—"))
        col3.metric("Stations",      len(stations))
        col4.metric("Total Sensors", len(all_sensors))

        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Time-Series Sensors", type_counts.get("time_series", 0))
        tc2.metric("Scalar Sensors",      type_counts.get("scalar", 0))
        tc3.metric("Binary Sensors",      type_counts.get("binary", 0))
        tc4.metric("Categorical Sensors", type_counts.get("categorical", 0))

        st.divider()
        st.markdown("#### Assembly Line Flow")
        _diagram_path = Path(__file__).parent.parent / "outputs" / "diagrams" / "06_plant_station_config.png"
        st.image(str(_diagram_path), use_container_width=True)

        ordered_stns = sorted(stations, key=lambda s: s.get("station_order", 0))

        st.divider()
        st.markdown("#### Station Details")
        station_rows = []
        for stn in ordered_stns:
            station_rows.append({
                "Order":         stn.get("station_order", "—"),
                "Station UID":   stn["station_uid"],
                "Name":          stn["station_name"],
                "Type":          stn["station_type"],
                "Capacity / hr": stn.get("capacity_per_hour", "—"),
                "Connected To":  ", ".join(stn.get("connected_to", [])) or "—  (end of line)",
            })
        st.dataframe(pd.DataFrame(station_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Sensor Specifications")
        sensor_rows = []
        for stn in ordered_stns:
            for sensor in stn.get("sensors", []):
                sensor_rows.append({
                    "Sensor UID":  sensor["sensor_uid"],
                    "Sensor Name": sensor.get("sensor_name", "—"),
                    "Station":     stn["station_uid"],
                    "Data Type":   sensor.get("data_type", "—"),
                    "Category":    sensor.get("sensor_category", "—"),
                    "Unit":        sensor.get("unit", "—"),
                    "Lower Limit": sensor.get("lower_limit", "—"),
                    "Upper Limit": sensor.get("upper_limit", "—"),
                    "Sampling Hz": sensor.get("sampling_hz", "—"),
                })
        st.dataframe(pd.DataFrame(sensor_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### Warranty Risk Scoring Weights")
        st.caption(
            "Weights used in the stochastic sigmoid label generation. "
            "Higher weight = stronger predictor of in-warranty claims."
        )
        weight_rows = [
            {"Sensor UID": "SNS_VISUAL_DEFECT_001", "Sensor Name": "Visual Defect Detection",
             "Risk Weight": 3.0,        "Rationale": "Visible defects — strongest warranty signal"},
            {"Sensor UID": "SNS_CONTINUITY_001",    "Sensor Name": "Wiring Continuity",
             "Risk Weight": 2.5,        "Rationale": "Wiring failure → electrical warranty risk"},
            {"Sensor UID": "SNS_VIBRATION_001",     "Sensor Name": "Vehicle Vibration",
             "Risk Weight": 2.0,        "Rationale": "High vibration → mechanical wear"},
            {"Sensor UID": "SNS_TORQUE_001",        "Sensor Name": "Wheel Torque",
             "Risk Weight": 1.5,        "Rationale": "Torque out-of-spec → assembly defect"},
            {"Sensor UID": "SNS_WELD_TEMP_001",     "Sensor Name": "Welding Temperature",
             "Risk Weight": 1.2,        "Rationale": "Weld temp drift → structural quality"},
            {"Sensor UID": "SNS_PAINT_THICK_001",   "Sensor Name": "Paint Thickness",
             "Risk Weight": "Excluded", "Rationale": "Cosmetic — low correlation with claims"},
        ]
        st.dataframe(pd.DataFrame(weight_rows), use_container_width=True, hide_index=True)

    # ─── TAB 2: Simulation ────────────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Manufacturing Simulation")
        if pipe is None:
            st.info("Run the pipeline using the sidebar.")
        else:
            vdf = pipe.vehicle_df
            mdf = pipe.manufacturing_df

            col1, col2, col3 = st.columns(3)
            col1.metric("Vehicles simulated", len(vdf))
            col2.metric("Sensor records",     f"{len(mdf):,}")
            col3.metric("Stations",           len(pipe.config.get("stations", [])))

            st.divider()
            st.markdown("#### Vehicle Metadata Summary")

            prod_min   = vdf["production_timestamp"].min()
            prod_max   = vdf["production_timestamp"].max()
            shift_dist = vdf["shift_id"].value_counts().sort_index().to_dict()
            env_count  = int(vdf["env_stress"].sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("Production Window",
                      f"{prod_min.strftime('%Y-%m-%d')} → {prod_max.strftime('%Y-%m-%d')}")
            m2.metric("Env. Stress Events",
                      f"{env_count}  ({env_count / len(vdf) * 100:.1f}% of fleet)")
            m3.metric("Production Interval", "1 vehicle every 5 minutes")

            s1, s2, s3 = st.columns(3)
            s1.metric("Morning Shift (0)",   shift_dist.get(0, 0))
            s2.metric("Afternoon Shift (1)", shift_dist.get(1, 0))
            s3.metric("Night Shift (2)",     shift_dist.get(2, 0))

            st.markdown("**Vehicle metadata (first 10 rows)**")
            st.dataframe(vdf.head(10), use_container_width=True)

            st.divider()
            st.markdown("#### Simulation Model Details")
            st.markdown("""
**AR(1) Autocorrelated Signal** — time-series sensors:
```
value[t] = PHI × value[t-1] + (1 − PHI) × (nominal + drift) + noise
PHI = 0.85  →  85% autocorrelation (smooth, physically realistic signal)
```

**Plant Aging / Drift Model:**
- `drift_factor = vehicle_index / N_vehicles`  → rises from 0.0 to ~1.0 over the run
- Binary failure rate climbs from **3% → 15%** as the plant ages
- Categorical "ok" rate falls from **90% → 70%**

**Environmental & Shift Effects:**
- `env_drift = env_stress × 0.20`  → power-outage vehicles get +20% extra drift
- `shift_noise = 1 + (night_shift) × 0.10`  → night crew produces 10% noisier readings

**Variable-Length Sequences (time-series sensors only):**
- Duration sampled uniformly: **1–3 seconds** per station pass
- Padded to uniform length for ML (see Prediction tab)
""")

            st.divider()
            st.markdown("#### Sensor Record Sample (first 20 rows)")
            st.dataframe(mdf.head(20), use_container_width=True)

            st.divider()
            st.markdown("#### Sensor Value Statistics")

            mdf_num = mdf.copy()
            mdf_num["value"] = pd.to_numeric(mdf_num["value"], errors="coerce")

            num_df = mdf_num[mdf_num["sensor_type"].isin(["time_series", "scalar"])]
            if not num_df.empty:
                st.markdown("**Numeric Sensors (time_series / scalar)**")
                numeric_stats = (
                    num_df.groupby("sensor_uid")["value"]
                    .agg(["mean", "std", "min", "max", "count"])
                    .round(3)
                    .rename(columns={"count": "readings"})
                )
                st.dataframe(numeric_stats, use_container_width=True)

            bin_df = mdf_num[mdf_num["sensor_type"] == "binary"]
            if not bin_df.empty:
                fail_rate = float(bin_df["value"].mean())
                b1, b2 = st.columns(2)
                b1.metric("Binary Sensor (Wiring Continuity) — Failure Rate",
                          f"{fail_rate:.3f}  ({fail_rate * 100:.1f}%)")
                b2.metric("Total Binary Readings", f"{len(bin_df):,}")

            cat_df = mdf[mdf["sensor_type"] == "categorical"]
            if not cat_df.empty:
                st.markdown("**Categorical Sensor (Visual Defect) — Value Distribution**")
                cat_counts = cat_df["value"].value_counts().reset_index()
                cat_counts.columns = ["Category", "Count"]
                cat_counts["Percentage"] = (
                    cat_counts["Count"] / len(cat_df) * 100
                ).round(1).astype(str) + "%"

                c_data, c_chart = st.columns([1, 2])
                with c_data:
                    st.dataframe(cat_counts, use_container_width=True, hide_index=True)
                with c_chart:
                    fig_cat, ax_cat = plt.subplots(figsize=(3.2, 2.0))
                    colors_cat = {"ok": "#2ECC71", "minor": "#F39C12", "major": "#E74C3C"}
                    bars_cat   = ax_cat.bar(
                        cat_counts["Category"],
                        cat_counts["Count"],
                        color=[colors_cat.get(str(c).lower(), "#95A5A6")
                               for c in cat_counts["Category"]],
                    )
                    ax_cat.set_title("Visual Defect Distribution", fontsize=9)
                    ax_cat.set_ylabel("Count")
                    for bar, cnt in zip(bars_cat, cat_counts["Count"]):
                        ax_cat.text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + 20, str(cnt),
                                    ha="center", va="bottom", fontsize=7)
                    fig_cat.tight_layout()
                    st.pyplot(fig_cat)
                    plt.close(fig_cat)

    # ─── TAB 3: Labels ────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("Warranty Claim Labels (Stochastic Sigmoid)")
        if pipe is None:
            st.info("Run the pipeline using the sidebar.")
        else:
            meta = pipe.label_meta
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total claims",      meta["claim_count"])
            col2.metric("Claim rate",        f"{meta['claim_rate']:.1%}")
            col3.metric("Sigmoid threshold", f"{meta['sigmoid_threshold']:.2f}")
            col4.metric("Max p_claim",       f"{meta['p_claim_max']:.3f}")

            st.markdown("**Feature matrix with warranty label (first 10 rows)**")
            display_df = pipe.feature_df_pre_opt.copy()
            if "warranty_claim" in display_df.columns:
                cols = ["warranty_claim"] + [c for c in display_df.columns if c != "warranty_claim"]
                display_df = display_df[cols]
            st.dataframe(display_df.head(10).round(3), use_container_width=True)

            fig, ax = plt.subplots(figsize=(4.0, 2.4))
            vc = pipe.feature_df_pre_opt["warranty_claim"].value_counts()
            ax.bar(["No Claim (0)", "Claim (1)"], [vc.get(0, 0), vc.get(1, 0)],
                   color=["#2ECC71", "#E74C3C"])
            ax.set_ylabel("Count")
            ax.set_title("Warranty Claim Distribution")
            fig.tight_layout()
            col_img, _ = st.columns([1, 1])
            with col_img:
                st.pyplot(fig)
            plt.close(fig)

    # ─── TAB 4: Data Preprocessing ────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Data Preprocessing")
        if pipe is None:
            st.info("Run the pipeline using the sidebar.")
        elif st.session_state.sim_preprocess_report is None:
            st.info("Preprocessing report not available. Re-run the pipeline.")
        else:
            st.caption(
                "Preprocessing is applied to the labeled feature matrix before "
                "tolerance optimization. For simulated data the report will typically "
                "show a clean result — this step becomes critical for real uploaded data."
            )
            _render_preprocess_report(
                report=st.session_state.sim_preprocess_report,
                df_before=pipe.feature_df_pre_opt,
                df_after=pipe.feature_df_pre_opt,
            )

    # ─── TAB 5: Tolerance Optimization ────────────────────────────────────────
    with tabs[4]:
        if pipe is None:
            st.info("Run the pipeline using the sidebar.")
        else:
            _render_ga_tab(pipe.ga_result, pipe.feature_df, pipe.c_fa)

    # ─── TAB 6: Prediction ────────────────────────────────────────────────────
    with tabs[5]:
        if pipe is None:
            st.info("Run the pipeline using the sidebar.")
        else:
            _render_prediction_tab(pipe.train_result, pipe.results_df)

    # ─── TAB 7: LIME Explanation ──────────────────────────────────────────────
    with tabs[6]:
        if pipe is None:
            st.info("Run the pipeline using the sidebar.")
        else:
            _render_lime_tab(pipe.lime_result, pipe.global_lime_result, pipe.train_result)


# ── Tabs — Upload mode ────────────────────────────────────────────────────────
else:
    ext  = st.session_state.ext
    tabs = st.tabs([
        "1 · Data Ingestion",
        "2 · Data Preprocessing",
        "3 · Tolerance Optimization",
        "4 · Prediction",
        "5 · LIME Explanation",
    ])

    # ─── TAB 1: Data Ingestion ────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Data Ingestion")
        st.caption(
            "Upload a CSV or Excel file containing your sensor / feature data "
            "plus a warranty claim label column. All preprocessing happens automatically "
            "in the next tab after you click Run Pipeline."
        )

        # on_change callback ensures session state is updated BEFORE the sidebar
        # re-renders, so the Run Pipeline button reflects the correct disabled state.
        st.file_uploader(
            "Upload data file",
            type=["csv", "xlsx", "xls"],
            key="_data_file",
            on_change=_on_file_upload,
            help="Supported formats: CSV, Excel (.xlsx, .xls)",
        )

        raw_df = ext.get("raw_df")

        if raw_df is not None:
            # ── Basic quality snapshot ────────────────────────────────────────
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Rows",                f"{len(raw_df):,}")
            q2.metric("Columns",             raw_df.shape[1])
            q3.metric("Total missing cells", f"{raw_df.isnull().sum().sum():,}")
            q4.metric("Duplicate rows",      f"{int(raw_df.duplicated().sum()):,}")

            st.divider()

            # ── Column type summary ───────────────────────────────────────────
            st.markdown("#### Column Overview")
            col_info = []
            for col in raw_df.columns:
                n_null = int(raw_df[col].isnull().sum())
                col_info.append({
                    "Column":        col,
                    "Dtype":         str(raw_df[col].dtype),
                    "Non-null":      int(raw_df[col].notna().sum()),
                    "Nulls":         n_null,
                    "Null %":        f"{n_null / len(raw_df) * 100:.1f}%",
                    "Unique":        int(raw_df[col].nunique()),
                    "Sample values": str(raw_df[col].dropna().unique()[:3].tolist()),
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

            st.divider()

            # ── Label column selector ─────────────────────────────────────────
            st.markdown("#### Select Label Column")
            st.caption(
                "Choose the column that contains the warranty claim label "
                "(must be binary: 0 = no claim, 1 = claim, or similar)."
            )

            # Compute default index from auto-detected label col
            current_label = ext.get("label_col")
            cols_list = list(raw_df.columns)
            default_idx = cols_list.index(current_label) if current_label in cols_list else 0

            st.selectbox(
                "Label column",
                options=cols_list,
                index=default_idx,
                key="_label_col",
                on_change=_on_label_select,
            )

            selected_label = ext.get("label_col") or cols_list[default_idx]

            # Show class balance for selected label
            try:
                vc = raw_df[selected_label].value_counts().sort_index()
                st.markdown(f"**Class balance for `{selected_label}`:**")
                balance_df = vc.reset_index()
                balance_df.columns = ["Value", "Count"]
                balance_df["Percentage"] = (
                    balance_df["Count"] / len(raw_df) * 100
                ).round(1).astype(str) + "%"
                cl1, cl2 = st.columns([1, 2])
                with cl1:
                    st.dataframe(balance_df, use_container_width=True, hide_index=True)
                with cl2:
                    fig_bal, ax_bal = plt.subplots(figsize=(3.2, 2.4))
                    x_labels = balance_df["Value"].astype(str).tolist()
                    ax_bal.bar(
                        x_labels,
                        balance_df["Count"],
                        color=["#2ECC71", "#E74C3C"] + ["#95A5A6"] * max(0, len(balance_df) - 2),
                    )
                    short_title = (selected_label[:25] + "…") if len(selected_label) > 25 else selected_label
                    ax_bal.set_title(f"Label: {short_title}", fontsize=9)
                    ax_bal.set_ylabel("Count", fontsize=8)
                    ax_bal.tick_params(axis="x", labelsize=8, rotation=45)
                    ax_bal.tick_params(axis="y", labelsize=8)
                    fig_bal.tight_layout()
                    st.pyplot(fig_bal)
                    plt.close(fig_bal)
            except Exception:
                pass

            st.divider()
            st.markdown("#### Raw Data Preview (first 10 rows)")
            st.dataframe(raw_df.head(10), use_container_width=True)

            st.info(
                f"Label column set to **`{selected_label}`**. "
                "Adjust GA settings in the sidebar then click **Run Pipeline** to proceed."
            )
        else:
            st.info("Upload a CSV or Excel file to get started.")

    # ─── TAB 2: Data Preprocessing ────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Data Preprocessing")
        if not ext.get("preprocess_report"):
            st.info("Run the pipeline using the sidebar to see the preprocessing report.")
        else:
            _render_preprocess_report(
                report=ext["preprocess_report"],
                df_before=ext["raw_df"],
                df_after=ext["preprocessed_df"],
            )

    # ─── TAB 3: Tolerance Optimization ────────────────────────────────────────
    with tabs[2]:
        if not ext.get("ga_result"):
            st.info("Run the pipeline using the sidebar.")
        else:
            _render_ga_tab(ext["ga_result"], ext["feature_df"], ext["c_fa"])

    # ─── TAB 4: Prediction ────────────────────────────────────────────────────
    with tabs[3]:
        if not ext.get("train_result"):
            st.info("Run the pipeline using the sidebar.")
        else:
            _render_prediction_tab(ext["train_result"], ext["results_df"])

    # ─── TAB 5: LIME Explanation ──────────────────────────────────────────────
    with tabs[4]:
        if not ext.get("lime_result"):
            st.info("Run the pipeline using the sidebar.")
        else:
            _render_lime_tab(
                ext["lime_result"],
                ext["global_lime_result"],
                ext["train_result"],
            )
