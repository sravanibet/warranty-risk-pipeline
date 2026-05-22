"""
Auto-preprocessing for warranty risk pipeline.

Returns a cleaned DataFrame plus a step-by-step report of every action taken
so the UI can display exactly what happened (and why).
"""
import pandas as pd


def auto_preprocess(df: pd.DataFrame, label_col: str = None):
    """
    Clean df automatically.

    Returns
    -------
    (cleaned_df, report)

    report : list of dicts with keys
        step    — short label for the action
        detail  — human-readable description of what was done
        status  — "ok" | "warning" | "info"
    """
    df = df.copy()
    report = []
    orig_rows, orig_cols = df.shape

    report.append({
        "step": "Input shape",
        "detail": f"{orig_rows:,} rows × {orig_cols} columns",
        "status": "info",
    })

    # ── 1. Duplicate rows ─────────────────────────────────────────────────────
    n_dup = int(df.duplicated().sum())
    if n_dup:
        df = df.drop_duplicates()
        report.append({
            "step": "Duplicate rows",
            "detail": f"Removed {n_dup:,} duplicate rows → {len(df):,} rows remain",
            "status": "warning",
        })
    else:
        report.append({
            "step": "Duplicate rows",
            "detail": "None found",
            "status": "ok",
        })

    # ── 2. Missing values ─────────────────────────────────────────────────────
    missing = df.isnull().sum()
    miss_cols = missing[missing > 0]

    if miss_cols.empty:
        report.append({
            "step": "Missing values",
            "detail": "None found across all columns",
            "status": "ok",
        })
    else:
        cols_to_drop = []
        for col in list(miss_cols.index):
            if col not in df.columns:
                continue
            n = int(df[col].isnull().sum())
            pct = n / len(df) * 100

            if pct > 50:
                cols_to_drop.append(col)
                report.append({
                    "step": "Missing values",
                    "detail": f"'{col}': {n:,} missing ({pct:.0f}%) — column dropped (>50% missing)",
                    "status": "warning",
                })
            elif col == label_col:
                before = len(df)
                df = df.dropna(subset=[col])
                removed = before - len(df)
                report.append({
                    "step": "Missing values",
                    "detail": f"'{col}' (label): {removed:,} rows with missing label dropped",
                    "status": "warning",
                })
            elif pd.api.types.is_numeric_dtype(df[col]):
                med = df[col].median()
                df[col] = df[col].fillna(med)
                report.append({
                    "step": "Missing values",
                    "detail": f"'{col}': {n:,} missing → filled with median ({med:.4g})",
                    "status": "warning",
                })
            else:
                mode_vals = df[col].mode()
                fill = mode_vals[0] if not mode_vals.empty else "UNKNOWN"
                df[col] = df[col].fillna(fill)
                report.append({
                    "step": "Missing values",
                    "detail": f"'{col}': {n:,} missing → filled with mode ('{fill}')",
                    "status": "warning",
                })

        if cols_to_drop:
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # ── 3. Non-numeric feature columns ───────────────────────────────────────
    non_num = [
        c for c in df.columns
        if c != label_col and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_num:
        df = df.drop(columns=non_num)
        report.append({
            "step": "Non-numeric columns",
            "detail": f"Dropped {len(non_num)} non-numeric feature column(s): {non_num}",
            "status": "warning",
        })
    else:
        report.append({
            "step": "Non-numeric columns",
            "detail": "All feature columns are numeric",
            "status": "ok",
        })

    # ── 4. Constant / zero-variance columns ───────────────────────────────────
    const_cols = [
        c for c in df.columns
        if c != label_col and df[c].nunique(dropna=False) <= 1
    ]
    if const_cols:
        df = df.drop(columns=const_cols)
        report.append({
            "step": "Constant columns",
            "detail": f"Dropped {len(const_cols)} zero-variance column(s): {const_cols}",
            "status": "warning",
        })
    else:
        report.append({
            "step": "Constant columns",
            "detail": "No zero-variance columns found",
            "status": "ok",
        })

    # ── 5. Label column type coercion ─────────────────────────────────────────
    if label_col and label_col in df.columns:
        try:
            df[label_col] = df[label_col].astype(int)
            uniq = sorted(df[label_col].unique().tolist())
            report.append({
                "step": "Label column",
                "detail": f"'{label_col}' coerced to int. Unique values: {uniq}",
                "status": "ok",
            })
        except Exception as e:
            report.append({
                "step": "Label column",
                "detail": f"Could not coerce '{label_col}' to int: {e}",
                "status": "warning",
            })

    # ── 6. Output summary ─────────────────────────────────────────────────────
    final_rows, final_cols = df.shape
    rows_removed = orig_rows - final_rows
    cols_removed = orig_cols - final_cols
    report.append({
        "step": "Output shape",
        "detail": (
            f"{final_rows:,} rows × {final_cols} columns "
            f"({rows_removed:,} rows and {cols_removed} columns removed)"
        ),
        "status": "info",
    })

    return df, report
