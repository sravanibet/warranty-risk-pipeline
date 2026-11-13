# Quality Monitoring Agent — Streamlit UI
# Run: streamlit run warranty_risk_app/app.py
#
# Four phases, orchestrated by warranty_risk.agent.QualityMonitoringAgent:
#   1. Setup    — register + confirm the facility configuration
#   2. Data     — upload real data or simulate it
#   3. Learning — tolerance optimization + model training
#   4. Operate  — score vehicles and explain why
#
# Developed by: Noor-E Jannat, PhD Candidate
#               Industrial & Systems Engineering (ISE) Department
#               Wayne State University
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

from warranty_risk.agent import QualityMonitoringAgent
from warranty_risk.agent.orchestrator import SIMULATOR_AVAILABLE

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quality Monitoring Agent",
    page_icon="🏭",
    layout="wide",
)
st.title("Quality Monitoring Agent")
st.caption(
    "The agent orchestrates and explains each phase; the pipeline modules do the "
    "actual computation."
)

# ── Session state init ──────────────────────────────────────────────────────────
for key, default in [
    ("agent", QualityMonitoringAgent()),
    ("phase_setup_confirmed", False),
    ("phase_data_ready", False),
    ("phase_learn_done", False),
    ("data_result", None),
    ("learn_result", None),
    ("validation_result", None),
    ("operate_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

agent = st.session_state.agent


# ── Shared rendering helpers (unchanged from the original tab layout) ──────────

def _render_preprocess_report(report: list, df_before, df_after):
    # Render the auto-preprocessing report with before/after comparison.
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
    # Render Tolerance Optimization section content.
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
    # Render Model Training & Selection section content.
    from sklearn.metrics import f1_score, precision_score, recall_score

    st.subheader("Model Training & Selection (Tolerance-Enhanced Features)")
    tr = train_result
    st.markdown(
        f"**Best model:** `{tr['best_name']}` (selected by validation F1) · "
        f"threshold: `{tr['best_threshold']:.3f}` (chosen on validation)"
    )
    st.caption(
        f"Vehicles split before any fitting: {len(tr['X_train'])} train / "
        f"{len(tr['X_val'])} validation / {len(tr['X_test'])} test. Tolerance bounds were fit "
        "on train vehicles only. F1, Precision, Recall and AUC below are measured on the "
        "test vehicles, which played no part in fitting, model choice or the threshold."
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
    # Render LIME Explanation section content.
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


def _render_config_summary(config: dict, diagram_path):
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

    if diagram_path:
        st.divider()
        st.markdown("#### Facility Topology — as understood by the agent")
        st.image(str(diagram_path), use_container_width=True)

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
                "Risk Weight": sensor.get("risk_weight", "1.0 (default)"),
            })
    st.dataframe(pd.DataFrame(sensor_rows), use_container_width=True, hide_index=True)


# ── Phase 1: Setup ──────────────────────────────────────────────────────────────
st.header("1 · Setup")

with st.expander("Facility configuration", expanded=not st.session_state.phase_setup_confirmed):
    uploaded_config = st.file_uploader(
        "Upload a plant configuration JSON (or leave empty to use the default)",
        type=["json"], key="_config_file",
    )
    setup_btn = st.button("Register facility", type="primary")

    if setup_btn:
        import json as _json
        config_arg = _json.load(uploaded_config) if uploaded_config else None
        st.session_state.setup_result = agent.setup(config_arg)
        st.session_state.phase_setup_confirmed = False
        st.session_state.phase_data_ready = False
        st.session_state.phase_learn_done = False

    setup_result = getattr(agent, "setup_result", None)
    if setup_result is not None:
        st.info(setup_result.narration)

        if not setup_result.validation.is_valid:
            st.error("Configuration problems found:")
            for err in setup_result.validation.errors:
                st.markdown(f"- {err}")
        else:
            if setup_result.validation.warnings:
                for w in setup_result.validation.warnings:
                    st.warning(w)

            _render_config_summary(setup_result.config, setup_result.diagram_path)

            if not st.session_state.phase_setup_confirmed:
                if st.button("✅ Confirm this is my facility", type="primary"):
                    agent.confirm_setup()
                    st.session_state.phase_setup_confirmed = True
                    st.rerun()
            else:
                st.success("Facility confirmed.")

if not st.session_state.phase_setup_confirmed:
    st.stop()

# ── Phase 2: Data ────────────────────────────────────────────────────────────────
st.header("2 · Data")

with st.expander("Data source", expanded=not st.session_state.phase_data_ready):
    data_options = ["Upload real data"]
    if SIMULATOR_AVAILABLE:
        data_options.append("Simulate")
    else:
        st.caption(
            "Simulation is unavailable — the `plant_simulator` package isn't installed "
            "(`pip install -e ./plant_simulator`). Upload real data instead."
        )
    data_source = st.radio("Do you have real data?", data_options, horizontal=True)

    if data_source == "Simulate":
        n_vehicles = st.slider("Number of vehicles", 100, 1000, 500, step=50)
        if st.button("Simulate data", type="primary"):
            with st.spinner("Simulating..."):
                st.session_state.data_result = agent.data("simulate", n_vehicles=n_vehicles)
            st.session_state.phase_data_ready = (
                st.session_state.data_result.feature_df is not None
            )
            st.session_state.phase_learn_done = False

    else:
        data_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if data_file is not None:
            raw_df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)
            cols_list = list(raw_df.columns)
            label_col = st.selectbox("Label column (warranty claim)", options=cols_list)
            if st.button("Validate data", type="primary"):
                st.session_state.data_result = agent.data("upload", df=raw_df, label_col=label_col)
                st.session_state.phase_data_ready = (
                    st.session_state.data_result.upload_validation is not None
                    and st.session_state.data_result.upload_validation.is_valid
                )
                st.session_state.phase_learn_done = False

    dr = st.session_state.data_result
    if dr is not None:
        st.info(dr.narration)
        if dr.upload_validation is not None:
            uv = dr.upload_validation
            if uv.is_valid:
                _render_preprocess_report(uv.preprocess_report, uv.raw_df, uv.cleaned_df)
                st.warning(
                    "Tolerance optimization and model training currently require simulated "
                    "sensor time-series (uploaded tabular data only supports validation and "
                    "preprocessing preview for now)."
                )

if not st.session_state.phase_data_ready:
    st.stop()

# ── Phase 3: Learning ────────────────────────────────────────────────────────────
st.header("3 · Learning")

can_learn = agent.pipe is not None and agent.pipe.feature_df is not None

with st.expander("Tolerance optimization & model training", expanded=not st.session_state.phase_learn_done):
    if not can_learn:
        st.info("Learning requires simulated data (see note in the Data phase above).")
    else:
        n_generations = st.slider("GA generations",     10,  100,  50, step=10)
        pop_size      = st.slider("GA population size", 50,  500, 200, step=50)
        c_fa = st.number_input("Cost false-accept (C_FA)", value=0.1, step=0.05)
        c_fn = st.number_input("Cost false-reject (C_FN)", value=2.0, step=0.5)

        if st.button("Run learning", type="primary"):
            with st.spinner("Optimizing tolerances and training models..."):
                st.session_state.learn_result = agent.learn(
                    n_generations=n_generations, pop_size=pop_size, c_fa=c_fa, c_fn=c_fn,
                )
            st.session_state.phase_learn_done = True

        lr = st.session_state.learn_result
        if lr is not None:
            st.info(lr.narration)
            _render_ga_tab(lr.ga_result, agent.pipe.feature_df, c_fa)
            st.divider()
            _render_prediction_tab(lr.train_result, lr.results_df)

            if lr.weight_adjustments:
                st.divider()
                st.markdown("#### Self-Improvement — Sensor Weight Adjustments")
                st.caption(
                    "The agent compared each sensor's out-of-tolerance signal against real "
                    "claim outcomes this run, and nudged its risk weight toward what actually "
                    "predicted claims — so the next run scores risk a little more accurately."
                )
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Sensor": a.sensor_uid,
                            "Old Weight": a.old_weight,
                            "New Weight": a.new_weight,
                            "Correlation w/ Claims": a.correlation,
                        }
                        for a in lr.weight_adjustments
                    ]),
                    use_container_width=True, hide_index=True,
                )

            st.divider()
            st.markdown("#### Statistical Validation")
            st.caption(
                "A single training run's F1 score doesn't say how stable that number is. "
                "This retrains several times with different random seeds and compares against "
                "a trivial baseline (flag anything out of tolerance on any sensor)."
            )
            n_runs = st.slider("Validation runs", 3, 10, 5, key="_n_validate_runs")
            if st.button("Run statistical validation"):
                with st.spinner(f"Retraining {n_runs} times..."):
                    st.session_state.validation_result = agent.validate(n_runs=n_runs)

            vr = st.session_state.get("validation_result")
            if vr is not None:
                vc1, vc2, vc3, vc4 = st.columns(4)
                vc1.metric("F1 (mean ± std)", f"{vr.f1_mean:.3f} ± {vr.f1_std:.3f}")
                vc2.metric("Precision (mean ± std)", f"{vr.precision_mean:.3f} ± {vr.precision_std:.3f}")
                vc3.metric("Recall (mean ± std)", f"{vr.recall_mean:.3f} ± {vr.recall_std:.3f}")
                vc4.metric("AUC (mean ± std)", f"{vr.auc_mean:.3f} ± {vr.auc_std:.3f}")
                st.metric(
                    "Lift over trivial baseline (F1)",
                    f"{vr.lift_over_baseline:+.3f}",
                    help=f"Baseline F1 (out-of-tolerance-on-any-sensor rule): {vr.baseline_f1_mean:.3f}",
                )
                st.markdown("**Per-run results**")
                st.dataframe(pd.DataFrame(vr.per_run), use_container_width=True, hide_index=True)
                if len(set(r["best_name"] for r in vr.per_run)) > 1:
                    st.warning(
                        "The best model changed across runs — the current single-run selection "
                        "isn't necessarily stable. Treat the \"best model\" label with caution."
                    )

if not st.session_state.phase_learn_done:
    st.stop()

# ── Phase 4: Operate ─────────────────────────────────────────────────────────────
st.header("4 · Operate")

if st.button("Score highest-risk vehicle", type="primary"):
    with st.spinner("Scoring..."):
        st.session_state.operate_result = agent.operate()

opr = st.session_state.operate_result
if opr is not None:
    st.info(opr.narration)
    _render_lime_tab(opr.lime_result, opr.global_lime_result, agent.pipe.train_result)
else:
    st.info("Click above to score a vehicle and see the agent's recommendation.")

# ── Agent activity log ───────────────────────────────────────────────────────────
st.divider()
with st.expander("Agent activity log (audit trail)"):
    st.caption(
        "Every decision the agent has made this session, in order — persisted to "
        "outputs/agent_audit_log.jsonl."
    )
    if not agent.audit_log.entries:
        st.info("No actions logged yet.")
    else:
        st.dataframe(
            pd.DataFrame(agent.audit_log.entries),
            use_container_width=True, hide_index=True,
        )
