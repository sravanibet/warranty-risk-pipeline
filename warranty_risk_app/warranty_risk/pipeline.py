"""
WarrantyRiskPipeline — end-to-end orchestrator.

Usage
-----
    from warranty_risk.pipeline import WarrantyRiskPipeline

    pipe = WarrantyRiskPipeline()
    pipe.run()
    print(pipe.results_df)
    print(pipe.lime_result["lime_df"])
"""
from pathlib import Path

from .config import load_config, get_stations

try:
    from plant_simulator import generate_vehicle_metadata, generate_manufacturing_data
except ImportError:
    generate_vehicle_metadata = None
    generate_manufacturing_data = None

from .labels.features import build_feature_matrix
from .labels.sigmoid import generate_warranty_labels
from .optimization.ga_optimizer import run_ga_optimization
from .optimization.tolerance_features import compute_tolerance_features
from .prediction.padding import build_padded_matrix
from .prediction.models import train_and_select
from .explanation.lime_explainer import explain_vehicle, global_lime


class WarrantyRiskPipeline:
    """
    Runs the full pipeline and stores every intermediate artifact as
    an attribute so callers (Streamlit app, notebooks, tests) can inspect
    any stage without re-running it.
    """

    def __init__(
        self,
        config_path=None,
        n_vehicles: int = 500,
        n_generations: int = 50,
        pop_size: int = 200,
        c_fa: float = 0.1,
        c_fn: float = 2.0,
        verbose_ga: bool = False,
    ):
        self.config_path   = config_path
        self.n_vehicles    = n_vehicles
        self.n_generations = n_generations
        self.pop_size      = pop_size
        self.c_fa          = c_fa
        self.c_fn          = c_fn
        self.verbose_ga    = verbose_ga

        # Artifacts — populated by run()
        self.config              = None
        self.vehicle_df          = None
        self.manufacturing_df    = None
        self.feature_df          = None
        self.feature_df_pre_opt  = None   # feature_df before tolerance columns are added
        self.label_meta          = None
        self.ga_result           = None
        self.sensor_rename       = None
        self.sensor_rename_inv   = None
        self.sensor_cols         = None
        self.ml_data             = None
        self.train_result        = None
        self.results_df          = None   # model leaderboard
        self.lime_result         = None
        self.global_lime_result  = None

    # ------------------------------------------------------------------
    def run(self, explain_vehicle_idx: int = None) -> "WarrantyRiskPipeline":
        """
        Execute all pipeline stages in order.
        Pass explain_vehicle_idx to explain a specific test-set row index;
        defaults to the highest-risk vehicle.
        """
        self._stage_simulate()
        self._stage_labels()
        self._stage_optimize()
        self._stage_predict()
        self._stage_explain(explain_vehicle_idx)
        return self

    # ------------------------------------------------------------------
    def _stage_simulate(self):
        if generate_vehicle_metadata is None:
            raise RuntimeError(
                "The plant_simulator package is not installed. "
                "Install it with `pip install -e ./plant_simulator`, or upload real data instead."
            )
        if self.config is None:
            self.config = load_config(self.config_path)
        stations          = get_stations(self.config)
        self.vehicle_df   = generate_vehicle_metadata(n=self.n_vehicles)
        self.manufacturing_df = generate_manufacturing_data(self.vehicle_df, stations)

    def _stage_labels(self):
        self.feature_df, self.label_meta = generate_warranty_labels(
            build_feature_matrix(self.manufacturing_df),
            config=self.config,
        )

    def _stage_optimize(self):
        # Snapshot feature_df before tolerance columns are appended (for Tab 2 display)
        self.feature_df_pre_opt = self.feature_df.copy()

        sensor_cols = [c for c in self.feature_df.columns if c != "warranty_claim"]
        self.sensor_cols = sensor_cols

        sensor_rename     = {s: f"sensor_{i+1}" for i, s in enumerate(sensor_cols)}
        sensor_rename_inv = {v: k for k, v in sensor_rename.items()}
        self.sensor_rename     = sensor_rename
        self.sensor_rename_inv = sensor_rename_inv

        self.ga_result = run_ga_optimization(
            feature_df=self.feature_df,
            sensor_cols=sensor_cols,
            sensor_rename=sensor_rename,
            sensor_rename_inv=sensor_rename_inv,
            n_generations=self.n_generations,
            pop_size=self.pop_size,
            c_fa=self.c_fa,
            c_fn=self.c_fn,
            verbose=self.verbose_ga,
        )

        if self.ga_result["success"]:
            self.feature_df = compute_tolerance_features(
                self.feature_df,
                self.ga_result["lowers_opt"],
                self.ga_result["uppers_opt"],
                self.ga_result["sensors_opt"],
                sensor_rename_inv,
            )

    def _stage_predict(self):
        self.ml_data      = build_padded_matrix(self.manufacturing_df, self.feature_df)
        self.train_result = train_and_select(self.ml_data)
        self.results_df   = self.train_result["results_df"]

    def _stage_explain(self, vehicle_idx: int = None):
        tr       = self.train_result
        best_name = tr["best_name"]

        from .prediction.models import SCALE_NEEDED
        needs_scale = best_name in SCALE_NEEDED
        X_tr = tr["X_train_scaled"] if needs_scale else tr["X_train"].values
        X_te = tr["X_test_scaled"]  if needs_scale else tr["X_test"].values
        feat_names = list(tr["feat_cols"])

        if vehicle_idx is None:
            probas = tr["best_model"].predict_proba(X_te)[:, 1]
            vehicle_idx = int(probas.argmax())

        self.lime_result = explain_vehicle(
            vehicle_idx=vehicle_idx,
            X_test=X_te,
            X_train=X_tr,
            feature_names=feat_names,
            predict_fn=tr["best_model"].predict_proba,
        )
        self.lime_result["vehicle_id"] = str(tr["X_test"].index[vehicle_idx])

        self.global_lime_result = global_lime(
            X_test=X_te,
            X_train=X_tr,
            y_test=tr["y_test"].values,
            feature_names=feat_names,
            predict_fn=tr["best_model"].predict_proba,
        )
