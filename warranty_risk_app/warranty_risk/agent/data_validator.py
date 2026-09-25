"""
Cross-checks uploaded data against the registered plant config: right sensors,
right types, nothing missing — before handing off to auto_preprocess.
"""
from dataclasses import dataclass, field

import pandas as pd

from ..preprocessing.preprocessor import auto_preprocess

_NUMERIC_TYPES = {"time_series", "scalar"}


@dataclass
class UploadValidationResult:
    missing_sensors: list = field(default_factory=list)
    extra_columns: list = field(default_factory=list)
    type_warnings: list = field(default_factory=list)
    preprocess_report: list = field(default_factory=list)
    raw_df: pd.DataFrame = None
    cleaned_df: pd.DataFrame = None

    @property
    def is_valid(self) -> bool:
        return not self.missing_sensors


def validate_upload(df: pd.DataFrame, config: dict, label_col: str) -> UploadValidationResult:
    expected = {}
    for station in config.get("stations", []):
        for sensor in station.get("sensors", []):
            uid = sensor.get("sensor_uid")
            if uid:
                expected[uid] = sensor.get("data_type")

    present_cols = set(df.columns)
    missing = [uid for uid in expected if uid not in present_cols]
    extra = [
        c for c in df.columns
        if c not in expected and c != label_col
    ]

    type_warnings = []
    for uid, dtype in expected.items():
        if uid not in present_cols:
            continue
        if dtype in _NUMERIC_TYPES and not pd.api.types.is_numeric_dtype(df[uid]):
            type_warnings.append(
                f"'{uid}' is registered as {dtype} but the uploaded column is not numeric."
            )

    cleaned_df, report = auto_preprocess(df, label_col=label_col)

    return UploadValidationResult(
        missing_sensors=missing,
        extra_columns=extra,
        type_warnings=type_warnings,
        preprocess_report=report,
        raw_df=df,
        cleaned_df=cleaned_df,
    )
