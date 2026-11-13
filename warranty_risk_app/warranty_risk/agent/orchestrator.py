# QualityMonitoringAgent — orchestrates the four phases described in plan.md:
# Setup, Data, Learning, Operate. It validates, narrates, and gates; the actual
# computation stays in the existing pipeline modules (WarrantyRiskPipeline and
# friends) which this class composes rather than reimplements.
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import load_config
from ..pipeline import WarrantyRiskPipeline
from .config_validator import validate_config, ValidationResult
from .diagram import render_topology
from .data_validator import validate_upload, UploadValidationResult
from .audit_log import AuditLog
from .feedback import compute_weight_adjustments, apply_weight_adjustments
from .narration import Narrator
from . import model_registry
from .validation import validate_learning, ValidationResult

try:
    import plant_simulator  # noqa: F401
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False


@dataclass
class SetupResult:
    config: dict
    validation: ValidationResult
    diagram_path: Path = None
    narration: str = ""


@dataclass
class DataResult:
    source: str
    narration: str
    upload_validation: UploadValidationResult = None
    feature_df: pd.DataFrame = None
    label_meta: dict = None


@dataclass
class LearnResult:
    narration: str
    ga_result: dict = None
    train_result: dict = None
    results_df: pd.DataFrame = None
    weight_adjustments: list = field(default_factory=list)


@dataclass
class OperateResult:
    narration: str
    vehicle_id: str = None
    probability: float = None
    recommendation: str = None
    lime_result: dict = None
    global_lime_result: dict = None


class QualityMonitoringAgent:
    # Wraps a WarrantyRiskPipeline instance and drives it phase by phase,
    # adding validation and plain-language narration at each step.

    def __init__(self):
        self.pipe: WarrantyRiskPipeline = None
        self.setup_result: SetupResult = None
        self.setup_confirmed = False
        self.audit_log = AuditLog()
        self.narrator = Narrator()

    # ------------------------------------------------------------------
    # Phase 1: Setup
    # ------------------------------------------------------------------
    def setup(self, config_path_or_dict=None, **pipeline_kwargs) -> SetupResult:
        if isinstance(config_path_or_dict, dict):
            config = config_path_or_dict
            config_path = None
        elif config_path_or_dict:
            config_path = Path(config_path_or_dict)
            with open(config_path) as f:
                config = json.load(f)
        else:
            config_path = None
            config = load_config()

        validation = validate_config(config)

        diagram_path = None
        narration_parts = []
        if not validation.is_valid:
            narration_parts.append(
                f"I found {len(validation.errors)} problem(s) with this configuration. "
                "Nothing will proceed until these are fixed."
            )
        else:
            diagram_path = render_topology(config)
            n_stations = len(config.get("stations", []))
            n_sensors = sum(len(s.get("sensors", [])) for s in config.get("stations", []))
            narration_parts.append(
                f"I registered '{config.get('plant_name', 'this facility')}' with "
                f"{n_stations} station(s) and {n_sensors} sensor(s). "
                "Here is the line as I understood it — confirm it matches your facility "
                "before I go any further."
            )
            if validation.warnings:
                narration_parts.append(
                    f"({len(validation.warnings)} non-blocking warning(s) — see below.)"
                )

        self.pipe = WarrantyRiskPipeline(config_path=config_path, **pipeline_kwargs)
        self.pipe.config = config  # let downstream stages see it even before run()

        fallback_narration = " ".join(narration_parts)
        narration = self.narrator.narrate(
            phase="setup",
            facts={
                "plant_name": config.get("plant_name"),
                "n_stations": len(config.get("stations", [])),
                "n_sensors": sum(len(s.get("sensors", [])) for s in config.get("stations", [])),
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "warnings": validation.warnings,
            },
            fallback=fallback_narration,
        )

        self.setup_result = SetupResult(
            config=config,
            validation=validation,
            diagram_path=diagram_path,
            narration=narration,
        )
        self.setup_confirmed = False
        self.audit_log.record(
            phase="setup", action="register_config",
            detail=self.setup_result.narration,
            valid=validation.is_valid,
        )
        return self.setup_result

    def confirm_setup(self):
        if self.setup_result is None or not self.setup_result.validation.is_valid:
            raise RuntimeError("Cannot confirm setup: no valid configuration registered yet.")
        self.setup_confirmed = True
        self.audit_log.record(
            phase="setup", action="confirm",
            detail=f"User confirmed facility '{self.setup_result.config.get('plant_name')}'.",
        )

    # ------------------------------------------------------------------
    # Phase 2: Data
    # ------------------------------------------------------------------
    def data(self, source: str, df: pd.DataFrame = None, label_col: str = None,
             n_vehicles: int = 500) -> DataResult:
        if not self.setup_confirmed:
            raise RuntimeError("Confirm the facility setup before providing data.")

        if source == "simulate":
            if not SIMULATOR_AVAILABLE:
                return DataResult(
                    source=source,
                    narration=(
                        "No plant_simulator package is installed, so I can't simulate data. "
                        "Install it with `pip install -e ./plant_simulator`, or upload real data instead."
                    ),
                )
            self.pipe.n_vehicles = n_vehicles
            self.pipe._stage_simulate()
            self.pipe._stage_labels()
            fallback_narration = (
                f"Simulated {n_vehicles} vehicles through the registered line. "
                f"Claim rate came out to {self.pipe.label_meta['claim_rate']:.1%}."
            )
            narration = self.narrator.narrate(
                phase="data",
                facts={
                    "source": "simulate",
                    "n_vehicles": n_vehicles,
                    "claim_rate": self.pipe.label_meta["claim_rate"],
                    "claim_count": self.pipe.label_meta["claim_count"],
                },
                fallback=fallback_narration,
            )
            self.audit_log.record(
                phase="data", action="simulate",
                detail=narration, n_vehicles=n_vehicles,
                claim_rate=self.pipe.label_meta["claim_rate"],
            )
            return DataResult(
                source=source,
                narration=narration,
                feature_df=self.pipe.feature_df,
                label_meta=self.pipe.label_meta,
            )

        elif source == "upload":
            if df is None or label_col is None:
                raise ValueError("Upload requires both a DataFrame and a label column.")
            validation = validate_upload(df, self.setup_result.config, label_col)
            if not validation.is_valid:
                fallback_narration = (
                    f"This data is missing {len(validation.missing_sensors)} sensor(s) "
                    f"registered in the plant config: {', '.join(validation.missing_sensors)}. "
                    "Upload data that covers every registered sensor before continuing."
                )
            else:
                fallback_narration = (
                    f"Data checks out against the registered configuration "
                    f"({len(df):,} rows). "
                )
                if validation.extra_columns:
                    fallback_narration += (
                        f"{len(validation.extra_columns)} column(s) aren't registered "
                        "sensors and were ignored. "
                    )
                if validation.type_warnings:
                    fallback_narration += " ".join(validation.type_warnings)
            narration = self.narrator.narrate(
                phase="data",
                facts={
                    "source": "upload",
                    "rows": len(df),
                    "is_valid": validation.is_valid,
                    "missing_sensors": validation.missing_sensors,
                    "extra_columns": validation.extra_columns,
                    "type_warnings": validation.type_warnings,
                },
                fallback=fallback_narration,
            )
            self.audit_log.record(
                phase="data", action="upload",
                detail=narration, rows=len(df), valid=validation.is_valid,
            )
            return DataResult(
                source=source,
                narration=narration,
                upload_validation=validation,
            )

        raise ValueError(f"Unknown data source: {source!r}")

    # ------------------------------------------------------------------
    # Phase 3: Learning
    # ------------------------------------------------------------------
    def learn(self, n_generations: int = 50, pop_size: int = 200,
              c_fa: float = 0.1, c_fn: float = 2.0) -> LearnResult:
        if self.pipe is None or self.pipe.feature_df is None:
            raise RuntimeError("No data available yet — complete the Data phase first.")

        self.pipe.n_generations = n_generations
        self.pipe.pop_size = pop_size
        self.pipe.c_fa = c_fa
        self.pipe.c_fn = c_fn

        self.pipe._stage_optimize()
        self.pipe._stage_predict()

        tr = self.pipe.train_result
        results_df = tr["results_df"]
        f1_val = float(results_df.loc[results_df["Model"] == tr["best_name"], "F1"].values[0])

        narration_parts = []
        if self.pipe.ga_result.get("success"):
            narration_parts.append(
                f"Tolerance optimization adjusted limits on "
                f"{len(self.pipe.ga_result.get('sensors_opt', []))} sensor(s)."
            )
        else:
            narration_parts.append(
                "Tolerance optimization did not find a valid solution — falling back to registered limits."
            )
        narration_parts.append(f"Best model: {tr['best_name']} (F1 = {f1_val:.3f}).")
        if f1_val < 0.5:
            narration_parts.append("This F1 score is low — treat these predictions with caution.")

        # Self-improvement: check how well each sensor's out-of-tolerance signal
        # actually tracked real claims, and nudge its risk_weight for next time.
        adjustments = compute_weight_adjustments(
            feature_df=self.pipe.feature_df,
            ga_result=self.pipe.ga_result,
            sensor_rename_inv=self.pipe.sensor_rename_inv,
            config=self.setup_result.config,
        )
        moved = []
        if adjustments:
            apply_weight_adjustments(self.setup_result.config, adjustments)
            moved = [a for a in adjustments if abs(a.new_weight - a.old_weight) >= 0.01]
            if moved:
                narration_parts.append(
                    f"Based on how well each sensor's out-of-tolerance signal matched real "
                    f"claims this run, I adjusted risk weights on {len(moved)} sensor(s) for "
                    f"next time: " + "; ".join(
                        f"{a.sensor_uid} {a.old_weight}→{a.new_weight}" for a in moved
                    ) + "."
                )

        fallback_narration = " ".join(narration_parts)
        narration = self.narrator.narrate(
            phase="learn",
            facts={
                "ga_success": self.pipe.ga_result.get("success"),
                "n_sensors_optimized": len(self.pipe.ga_result.get("sensors_opt", [])),
                "best_model": tr["best_name"],
                "f1": f1_val,
                "weight_adjustments": [
                    {"sensor": a.sensor_uid, "old": a.old_weight, "new": a.new_weight}
                    for a in moved
                ],
            },
            fallback=fallback_narration,
        )

        saved_path = model_registry.save(
            tr, f1_val,
            preprocessing={
                "category_map": self.pipe.trained_category_map,
                "sensor_uids": list(self.pipe.trained_sensor_uids),
                "max_seq_len": self.pipe.trained_max_seq_len,
            },
        )

        self.audit_log.record(
            phase="learn", action="train",
            detail=narration,
            best_model=tr["best_name"], f1=f1_val,
            weight_adjustments=[a.__dict__ for a in adjustments],
            model_saved_to=str(saved_path),
        )

        return LearnResult(
            narration=narration,
            ga_result=self.pipe.ga_result,
            train_result=self.pipe.train_result,
            results_df=self.pipe.results_df,
            weight_adjustments=adjustments,
        )

    def validate(self, n_runs: int = 5) -> ValidationResult:
        # Re-trains n_runs times with different seeds and reports mean +/- std
        # for F1/Precision/Recall/AUC, plus a trivial baseline comparison —
        # answers "how much do I trust this number" instead of reporting one
        # single-run metric as if it were exact.
        if self.pipe is None or self.pipe.ml_data is None:
            raise RuntimeError("No data available yet — complete Learning before validating.")

        result = validate_learning(self.pipe, n_runs=n_runs)

        self.audit_log.record(
            phase="learn", action="validate",
            detail=(
                f"Validated over {result.n_runs} runs: "
                f"F1 = {result.f1_mean:.3f} +/- {result.f1_std:.3f}, "
                f"baseline F1 = {result.baseline_f1_mean:.3f} "
                f"(lift = {result.lift_over_baseline:+.3f})."
            ),
            f1_mean=result.f1_mean, f1_std=result.f1_std,
            baseline_f1_mean=result.baseline_f1_mean,
        )
        return result

    # ------------------------------------------------------------------
    # Phase 4: Operate
    # ------------------------------------------------------------------
    def operate(self, vehicle_idx: int = None) -> OperateResult:
        if self.pipe is None or self.pipe.train_result is None:
            raise RuntimeError("No trained model yet — complete the Learning phase first.")

        self.pipe._stage_explain(vehicle_idx)
        return self._build_operate_result(
            self.pipe.lime_result, action="score_vehicle",
            global_lime_result=self.pipe.global_lime_result,
        )

    # ------------------------------------------------------------------
    # Live scoring: one new vehicle's readings, e.g. from an MQTT window,
    # scored against the already-trained model. This is the "Detection"
    # stage of the IntelliMake Listener -> Detection -> Alerting pipeline.
    # ------------------------------------------------------------------
    def score_vehicle_from_readings(self, readings_df: pd.DataFrame) -> OperateResult:
        if self.pipe is None or self.pipe.train_result is None:
            raise RuntimeError("No trained model yet — complete the Learning phase first.")

        lr = self.pipe.score_new_vehicle(readings_df)
        return self._build_operate_result(lr, action="score_live_vehicle")

    def _build_operate_result(self, lr: dict, action: str, global_lime_result: dict = None) -> OperateResult:
        proba = lr["proba"]

        if proba >= 0.7:
            recommendation = "High risk — inspect before shipment."
        elif proba >= 0.4:
            recommendation = "Elevated risk — flag for monitoring."
        else:
            recommendation = "Low risk — no action needed."

        fallback_narration = (
            f"Vehicle {lr['vehicle_id']} scored {proba:.1%} probability of a warranty claim. "
            f"{recommendation}"
        )
        top_features = lr["lime_df"].reindex(
            lr["lime_df"]["Weight"].abs().sort_values(ascending=False).index
        ).head(3)
        narration = self.narrator.narrate(
            phase="operate",
            facts={
                "vehicle_id": lr["vehicle_id"],
                "probability": proba,
                "recommendation": recommendation,
                "top_contributing_features": top_features[["Feature", "Weight"]].to_dict("records"),
            },
            fallback=fallback_narration,
        )

        self.audit_log.record(
            phase="operate", action=action,
            detail=narration, vehicle_id=lr["vehicle_id"], probability=proba,
            recommendation=recommendation,
        )

        return OperateResult(
            narration=narration,
            vehicle_id=lr["vehicle_id"],
            probability=proba,
            recommendation=recommendation,
            lime_result=lr,
            global_lime_result=global_lime_result,
        )
