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


def recommended_min_observations(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    frequency: str,
) -> int:
    """Set a reliability threshold from the available paired observations.

    The threshold is never optimized for the largest correlation coefficient,
    because doing so would favor noisy small samples. It requires at least half
    of the smallest usable series, subject to a 60-day or 12-month floor.
    """
    floors = {"Daily": 60, "Monthly": 12}
    if frequency not in floors:
        raise ValueError("frequency must be 'Daily' or 'Monthly'")

    counts = [
        len(_paired_numeric(df, target_col, feature_col)) for feature_col in feature_cols
    ]
    usable_counts = [count for count in counts if count > 0]
    if not usable_counts:
        return floors[frequency]

    available = min(usable_counts)
    return min(available, max(floors[frequency], int(np.ceil(available / 2))))


def lead_lag_correlations(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    max_lag: int = 12,
    min_periods: int = 6,
    lag_column: str = "lag_months",
) -> pd.DataFrame:
    """Calculate correlations after shifting each feature across a lag window."""
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
                    lag_column: lag,
                    "pearson": pearson,
                    "observations": observations,
                }
            )
    return pd.DataFrame.from_records(records)


def best_lag_summary(
    lag_df: pd.DataFrame,
    lag_column: str = "lag_months",
    best_lag_column: str = "best_lag_months",
) -> pd.DataFrame:
    """Select the strongest absolute Pearson correlation for each series."""
    if lag_df.empty:
        return pd.DataFrame(columns=["series", best_lag_column, "best_lag_pearson", "observations"])

    rows = lag_df.dropna(subset=["pearson"]).copy()
    if rows.empty:
        return pd.DataFrame(columns=["series", best_lag_column, "best_lag_pearson", "observations"])

    rows["abs_pearson"] = rows["pearson"].abs()
    best = rows.sort_values(["series", "abs_pearson"], ascending=[True, False])
    best = best.groupby("series", as_index=False).first()
    return best.rename(
        columns={
            lag_column: best_lag_column,
            "pearson": "best_lag_pearson",
        }
    )[["series", best_lag_column, "best_lag_pearson", "observations"]]


def top_lag_relationships(
    lag_df: pd.DataFrame,
    lag_column: str,
    limit: int = 10,
) -> pd.DataFrame:
    """Rank the strongest valid lead/lag relationships across all series."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    if lag_df.empty:
        return lag_df.copy()

    rows = lag_df.dropna(subset=["pearson"]).copy()
    rows["abs_pearson"] = rows["pearson"].abs()
    return (
        rows.sort_values(
            ["abs_pearson", "observations", "series", lag_column],
            ascending=[False, False, True, True],
        )
        .head(limit)
        .drop(columns="abs_pearson")
        .reset_index(drop=True)
    )
