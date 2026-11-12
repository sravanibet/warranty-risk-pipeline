"""
Preprocessing for the warranty risk pipeline.

`Preprocessor` follows the fit / transform pattern: everything it *learns*
(which columns to drop, which median to impute) comes from the data passed to
`fit` only, and `transform` re-applies exactly that to any other data. Fit it
on training vehicles and transform validation / test / live rows with it, so no
information from held-out data leaks into the imputation or column selection.

`auto_preprocess` keeps the original one-shot interface (fit and transform on
the same DataFrame, returning a step-by-step report for the UI).
"""
import pandas as pd

_TRUE = {"1", "true", "yes", "y", "t"}
_FALSE = {"0", "false", "no", "n", "f"}


def _coerce_label(series: pd.Series) -> pd.Series:
    """Cast a label column to int, accepting 0/1, bool and common yes/no strings."""
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    if not lowered.isin(_TRUE | _FALSE).all():
        bad = sorted(set(lowered[~lowered.isin(_TRUE | _FALSE)]))[:5]
        raise ValueError(f"unrecognised label values {bad}")
    return lowered.isin(_TRUE).astype(int)


class Preprocessor:
    def __init__(self, label_col: str = None, max_missing_frac: float = 0.5):
        self.label_col = label_col
        self.max_missing_frac = max_missing_frac
        self.fitted_ = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Learn drop-lists and imputation values from df (use training data only)."""
        df = df.drop_duplicates()
        self.fit_report_ = []
        self.dropped_missing_ = {}      # col -> (n_missing, pct)
        self.dropped_non_numeric_ = []
        self.dropped_constant_ = []
        self.fill_values_ = {}          # col -> (median, n_missing_at_fit)
        self.label_missing_rows_ = 0

        feature_cols = [c for c in df.columns if c != self.label_col]

        for col in feature_cols:
            n = int(df[col].isnull().sum())
            if n and n / len(df) > self.max_missing_frac:
                self.dropped_missing_[col] = (n, n / len(df) * 100)
        feature_cols = [c for c in feature_cols if c not in self.dropped_missing_]

        # A label that is mostly missing is a data problem to report, never a column to drop.
        if self.label_col in df.columns:
            self.label_missing_rows_ = int(df[self.label_col].isnull().sum())

        self.dropped_non_numeric_ = [
            c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])
        ]
        feature_cols = [c for c in feature_cols if c not in self.dropped_non_numeric_]

        for col in feature_cols:
            n = int(df[col].isnull().sum())
            if n:
                self.fill_values_[col] = (float(df[col].median()), n)

        filled = df[feature_cols].fillna({c: v[0] for c, v in self.fill_values_.items()})
        self.dropped_constant_ = [c for c in feature_cols if filled[c].nunique(dropna=False) <= 1]
        self.feature_cols_ = [c for c in feature_cols if c not in self.dropped_constant_]
        self.fitted_ = True
        return self

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame, drop_duplicates: bool = False,
                  strict_label: bool = True) -> pd.DataFrame:
        """
        Apply what fit learned. Rows are only removed for a missing label (and
        duplicates, when asked — do that for training data, not live rows).
        Raises if a column kept at fit time is absent, or (strict_label) if the
        label has values that can't be read as 0/1.
        """
        if not self.fitted_:
            raise RuntimeError("Preprocessor must be fit before transform.")
        missing = [c for c in self.feature_cols_ if c not in df.columns]
        if missing:
            raise ValueError(f"Data is missing column(s) the preprocessor was fit on: {missing}")

        df = df.copy()
        if drop_duplicates:
            df = df.drop_duplicates()

        has_label = self.label_col is not None and self.label_col in df.columns
        keep = self.feature_cols_ + ([self.label_col] if has_label else [])
        df = df[keep]
        df = df.fillna({c: v[0] for c, v in self.fill_values_.items() if c in df.columns})

        if has_label:
            df = df.dropna(subset=[self.label_col])
            try:
                df[self.label_col] = _coerce_label(df[self.label_col])
            except ValueError:
                if strict_label:
                    raise
        return df

    # --------------------------------------------------------- fit_transform
    def fit_transform(self, df: pd.DataFrame):
        """Fit and transform the same data. Returns (cleaned_df, report)."""
        self.fit(df)
        n_dup = int(df.duplicated().sum())
        cleaned = self.transform(df, drop_duplicates=True, strict_label=False)
        return cleaned, self._report(df, cleaned, n_dup)

    def _report(self, raw: pd.DataFrame, cleaned: pd.DataFrame, n_dup: int) -> list:
        r = []
        add = lambda step, detail, status: r.append({"step": step, "detail": detail, "status": status})

        add("Input shape", f"{raw.shape[0]:,} rows × {raw.shape[1]} columns", "info")

        if n_dup:
            add("Duplicate rows",
                f"Removed {n_dup:,} duplicate rows → {len(raw) - n_dup:,} rows remain", "warning")
        else:
            add("Duplicate rows", "None found", "ok")

        any_missing = False
        for col, (n, pct) in self.dropped_missing_.items():
            any_missing = True
            add("Missing values",
                f"'{col}': {n:,} missing ({pct:.0f}%) — column dropped "
                f"(>{self.max_missing_frac:.0%} missing)", "warning")
        for col, (med, n) in self.fill_values_.items():
            any_missing = True
            add("Missing values", f"'{col}': {n:,} missing → filled with median ({med:.4g})", "warning")
        if self.label_missing_rows_:
            any_missing = True
            add("Missing values",
                f"'{self.label_col}' (label): {self.label_missing_rows_:,} rows with missing label dropped",
                "warning")
        if not any_missing:
            add("Missing values", "None found across all columns", "ok")

        if self.dropped_non_numeric_:
            add("Non-numeric columns",
                f"Dropped {len(self.dropped_non_numeric_)} non-numeric feature column(s): "
                f"{self.dropped_non_numeric_}", "warning")
        else:
            add("Non-numeric columns", "All feature columns are numeric", "ok")

        if self.dropped_constant_:
            add("Constant columns",
                f"Dropped {len(self.dropped_constant_)} zero-variance column(s): "
                f"{self.dropped_constant_}", "warning")
        else:
            add("Constant columns", "No zero-variance columns found", "ok")

        if (self.label_col and self.label_col in cleaned.columns
                and pd.api.types.is_integer_dtype(cleaned[self.label_col])):
            uniq = sorted(cleaned[self.label_col].unique().tolist())
            add("Label column", f"'{self.label_col}' coerced to int. Unique values: {uniq}", "ok")
        elif self.label_col and self.label_col in raw.columns:
            add("Label column", f"Could not coerce '{self.label_col}' to int", "warning")

        add("Output shape",
            f"{cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns "
            f"({raw.shape[0] - cleaned.shape[0]:,} rows and {raw.shape[1] - cleaned.shape[1]} "
            f"columns removed)", "info")
        return r


def auto_preprocess(df: pd.DataFrame, label_col: str = None):
    """
    One-shot cleaning (fit and transform on the same data).

    Returns (cleaned_df, report); report is a list of dicts with keys
    step, detail, status ("ok" | "warning" | "info"). For anything that will be
    split into train/test, use Preprocessor: fit on train, transform the rest.
    """
    return Preprocessor(label_col=label_col).fit_transform(df)
