from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _month_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.to_period("M").dt.to_timestamp()


def monthly_volume(
    df: pd.DataFrame,
    date_col: str,
    value_col: str | None,
    output_col: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", output_col])
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: {date_col}")

    rows = df.copy()
    rows["month"] = _month_start(rows[date_col])
    rows = rows.dropna(subset=["month"])

    if value_col and value_col in rows.columns:
        rows[value_col] = pd.to_numeric(rows[value_col], errors="coerce")
        grouped = rows.groupby("month", as_index=False)[value_col].sum(min_count=1)
        grouped = grouped.rename(columns={value_col: output_col})
    else:
        grouped = rows.groupby("month").size().reset_index(name=output_col)

    return grouped.sort_values("month").reset_index(drop=True)


def monthly_baltic(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "p3a_82"])
    if date_col not in df.columns:
        raise KeyError(f"Missing required Baltic date column: {date_col}")
    if value_col not in df.columns:
        raise KeyError(f"Missing required Baltic value column: {value_col}")

    rows = df.copy()
    rows["month"] = _month_start(rows[date_col])
    rows[value_col] = pd.to_numeric(rows[value_col], errors="coerce")
    rows = rows.dropna(subset=["month", value_col])
    grouped = rows.groupby("month", as_index=False)[value_col].mean()
    return (
        grouped.rename(columns={value_col: "p3a_82"})
        .sort_values("month")
        .reset_index(drop=True)
    )


def build_monthly_dataset(
    baltic: pd.DataFrame,
    australia: pd.DataFrame,
    indonesia: pd.DataFrame,
    china: pd.DataFrame,
) -> pd.DataFrame:
    merged = baltic.copy()
    for frame in (australia, indonesia, china):
        merged = merged.merge(frame, on="month", how="inner")
    return merged.sort_values("month").reset_index(drop=True)


def add_indexed_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        first_valid = values.dropna().iloc[0] if not values.dropna().empty else np.nan
        output_col = f"{column}_index"
        if pd.isna(first_valid) or first_valid == 0:
            result[output_col] = np.nan
        else:
            result[output_col] = values / first_valid * 100.0
    return result


def add_change_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}_mom"] = values.pct_change(fill_method=None)
        result[f"{column}_yoy"] = values.pct_change(periods=12, fill_method=None)
    return result
