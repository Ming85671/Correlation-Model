from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _paired_numeric(
    df: pd.DataFrame,
    target_col: str,
    feature_col: str,
) -> pd.DataFrame:
    paired = df[[target_col, feature_col]].apply(pd.to_numeric, errors="coerce")
    return paired.dropna()


def _corr(paired: pd.DataFrame, target_col: str, feature_col: str, method: str) -> float:
    if method == "spearman":
        ranked = paired[[target_col, feature_col]].rank()
        value = ranked[target_col].corr(ranked[feature_col], method="pearson")
    else:
        value = paired[target_col].corr(paired[feature_col], method=method)
    return float(value) if pd.notna(value) else np.nan


def correlation_summary(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    min_periods: int = 6,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for feature_col in feature_cols:
        paired = _paired_numeric(df, target_col, feature_col)
        observations = len(paired)
        if observations < min_periods:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = _corr(paired, target_col, feature_col, "pearson")
            spearman = _corr(paired, target_col, feature_col, "spearman")

        records.append(
            {
                "series": feature_col,
                "pearson": pearson,
                "spearman": spearman,
                "observations": observations,
            }
        )
    return pd.DataFrame.from_records(records)


def lead_lag_correlations(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    max_lag: int = 12,
    min_periods: int = 6,
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for feature_col in feature_cols:
        for lag in range(-max_lag, max_lag + 1):
            shifted = df.copy()
            shifted[feature_col] = shifted[feature_col].shift(lag)
            paired = _paired_numeric(shifted, target_col, feature_col)
            observations = len(paired)
            pearson = (
                _corr(paired, target_col, feature_col, "pearson")
                if observations >= min_periods
                else np.nan
            )
            records.append(
                {
                    "series": feature_col,
                    "lag_months": lag,
                    "pearson": pearson,
                    "observations": observations,
                }
            )
    return pd.DataFrame.from_records(records)


def best_lag_summary(lag_df: pd.DataFrame) -> pd.DataFrame:
    if lag_df.empty:
        return pd.DataFrame(
            columns=["series", "best_lag_months", "best_lag_pearson", "observations"]
        )

    rows = lag_df.dropna(subset=["pearson"]).copy()
    if rows.empty:
        return pd.DataFrame(
            columns=["series", "best_lag_months", "best_lag_pearson", "observations"]
        )

    rows["abs_pearson"] = rows["pearson"].abs()
    best = rows.sort_values(["series", "abs_pearson"], ascending=[True, False])
    best = best.groupby("series", as_index=False).first()
    return best.rename(
        columns={
            "lag_months": "best_lag_months",
            "pearson": "best_lag_pearson",
        }
    )[["series", "best_lag_months", "best_lag_pearson", "observations"]]
