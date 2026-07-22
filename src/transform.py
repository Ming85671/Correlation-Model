from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _month_start(values: pd.Series) -> pd.Series:
    return (
        pd.to_datetime(values, errors="coerce", format="mixed")
        .dt.to_period("M")
        .dt.to_timestamp()
    )


def _day_start(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", format="mixed").dt.normalize()


def _week_start(values: pd.Series) -> pd.Series:
    """Return the Monday that starts each calendar week."""
    return (
        pd.to_datetime(values, errors="coerce", format="mixed")
        .dt.to_period("W-SUN")
        .dt.start_time
    )


def _coerce_volume(values: pd.Series) -> pd.Series:
    """Parse numeric cargo volumes stored with separators or common tonne units."""
    normalized = (
        values.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(r"(?i)\s*(?:mt|metric tons?|tonnes?|tons?)\s*$", "", regex=True)
    )
    return pd.to_numeric(normalized, errors="coerce")


def daily_volume(
    df: pd.DataFrame,
    date_col: str,
    value_col: str | None,
    output_col: str,
) -> pd.DataFrame:
    """Aggregate shipment or arrival records into calendar-day totals."""
    if df.empty:
        return pd.DataFrame(columns=["day", output_col])
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: {date_col}")

    rows = df.copy()
    rows["day"] = _day_start(rows[date_col])
    rows = rows.dropna(subset=["day"])

    if value_col and value_col in rows.columns:
        rows[value_col] = _coerce_volume(rows[value_col])
        grouped = rows.groupby("day", as_index=False)[value_col].sum(min_count=1)
        grouped = grouped.rename(columns={value_col: output_col})
    else:
        grouped = rows.groupby("day").size().reset_index(name=output_col)

    return grouped.sort_values("day").reset_index(drop=True)


def daily_flow_metrics(
    df: pd.DataFrame,
    date_col: str,
    value_col: str | None,
    count_output_col: str,
    volume_output_col: str,
) -> pd.DataFrame:
    """Aggregate daily shipment-record counts and cargo volumes separately."""
    if df.empty:
        return pd.DataFrame(columns=["day", count_output_col, volume_output_col])
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: {date_col}")

    rows = df.copy()
    rows["day"] = _day_start(rows[date_col])
    rows = rows.dropna(subset=["day"])
    grouped = rows.groupby("day").size().reset_index(name=count_output_col)
    if value_col and value_col in rows.columns:
        rows[value_col] = _coerce_volume(rows[value_col])
        volume = rows.groupby("day", as_index=False)[value_col].sum(min_count=1)
        grouped = grouped.merge(volume, on="day", how="left").rename(
            columns={value_col: volume_output_col}
        )
    else:
        grouped[volume_output_col] = np.nan
    return grouped.sort_values("day").reset_index(drop=True)


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
        rows[value_col] = _coerce_volume(rows[value_col])
        grouped = rows.groupby("month", as_index=False)[value_col].sum(min_count=1)
        grouped = grouped.rename(columns={value_col: output_col})
    else:
        grouped = rows.groupby("month").size().reset_index(name=output_col)

    return grouped.sort_values("month").reset_index(drop=True)


def monthly_flow_metrics(
    df: pd.DataFrame,
    date_col: str,
    value_col: str | None,
    count_output_col: str,
    volume_output_col: str,
) -> pd.DataFrame:
    """Aggregate monthly shipment-record counts and cargo volumes separately."""
    if df.empty:
        return pd.DataFrame(columns=["month", count_output_col, volume_output_col])
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: {date_col}")

    rows = df.copy()
    rows["month"] = _month_start(rows[date_col])
    rows = rows.dropna(subset=["month"])
    grouped = rows.groupby("month").size().reset_index(name=count_output_col)
    if value_col and value_col in rows.columns:
        rows[value_col] = _coerce_volume(rows[value_col])
        volume = rows.groupby("month", as_index=False)[value_col].sum(min_count=1)
        grouped = grouped.merge(volume, on="month", how="left").rename(
            columns={value_col: volume_output_col}
        )
    else:
        grouped[volume_output_col] = np.nan
    return grouped.sort_values("month").reset_index(drop=True)


def weekly_flow_metrics(
    df: pd.DataFrame,
    date_col: str,
    value_col: str | None,
    count_output_col: str,
    volume_output_col: str,
) -> pd.DataFrame:
    """Aggregate shipment-record counts and cargo volumes by Monday-start week."""
    if df.empty:
        return pd.DataFrame(columns=["week", count_output_col, volume_output_col])
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: {date_col}")

    rows = df.copy()
    rows["week"] = _week_start(rows[date_col])
    rows = rows.dropna(subset=["week"])
    grouped = rows.groupby("week").size().reset_index(name=count_output_col)
    if value_col and value_col in rows.columns:
        rows[value_col] = _coerce_volume(rows[value_col])
        volume = rows.groupby("week", as_index=False)[value_col].sum(min_count=1)
        grouped = grouped.merge(volume, on="week", how="left").rename(
            columns={value_col: volume_output_col}
        )
    else:
        grouped[volume_output_col] = np.nan
    return grouped.sort_values("week").reset_index(drop=True)


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


def weekly_baltic(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Average Baltic observations by Monday-start week."""
    if df.empty:
        return pd.DataFrame(columns=["week", "p3a_82"])
    if date_col not in df.columns:
        raise KeyError(f"Missing required Baltic date column: {date_col}")
    if value_col not in df.columns:
        raise KeyError(f"Missing required Baltic value column: {value_col}")

    rows = df.copy()
    rows["week"] = _week_start(rows[date_col])
    rows[value_col] = pd.to_numeric(rows[value_col], errors="coerce")
    rows = rows.dropna(subset=["week", value_col])
    grouped = rows.groupby("week", as_index=False)[value_col].mean()
    return (
        grouped.rename(columns={value_col: "p3a_82"})
        .sort_values("week")
        .reset_index(drop=True)
    )


def daily_baltic(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Aggregate Baltic observations into one mean value per calendar day."""
    if df.empty:
        return pd.DataFrame(columns=["day", "p3a_82"])
    if date_col not in df.columns:
        raise KeyError(f"Missing required Baltic date column: {date_col}")
    if value_col not in df.columns:
        raise KeyError(f"Missing required Baltic value column: {value_col}")

    rows = df.copy()
    rows["day"] = _day_start(rows[date_col])
    rows[value_col] = pd.to_numeric(rows[value_col], errors="coerce")
    rows = rows.dropna(subset=["day", value_col])
    grouped = rows.groupby("day", as_index=False)[value_col].mean()
    return (
        grouped.rename(columns={value_col: "p3a_82"})
        .sort_values("day")
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


def build_weekly_dataset(
    baltic: pd.DataFrame,
    australia: pd.DataFrame,
    indonesia: pd.DataFrame,
    china: pd.DataFrame,
) -> pd.DataFrame:
    """Align weekly Baltic and cargo metrics over their shared calendar weeks."""
    merged = baltic.copy()
    for frame in (australia, indonesia, china):
        merged = merged.merge(frame, on="week", how="inner")
    return merged.sort_values("week").reset_index(drop=True)


def build_daily_dataset(
    baltic: pd.DataFrame,
    australia: pd.DataFrame,
    indonesia: pd.DataFrame,
    china: pd.DataFrame,
) -> pd.DataFrame:
    """Align daily cargo flows to a complete calendar anchored on Baltic dates.

    Cargo-free days are meaningful observations, so they are retained as zeros.
    Baltic is left missing on weekends and market holidays rather than forward-filled.
    """
    flow_columns = list(
        dict.fromkeys(
            column
            for frame in (australia, indonesia, china)
            for column in frame.columns
            if column != "day"
        )
    )
    if baltic.empty:
        return pd.DataFrame(columns=["day", "p3a_82", *flow_columns])

    days = pd.date_range(baltic["day"].min(), baltic["day"].max(), freq="D")
    merged = pd.DataFrame({"day": days}).merge(baltic, on="day", how="left")
    for frame in (australia, indonesia, china):
        if frame.empty:
            for column in frame.columns:
                if column != "day":
                    merged[column] = np.nan
        else:
            merged = merged.merge(frame, on="day", how="left")
            for column in frame.columns:
                if column == "day" or not frame[column].notna().any():
                    continue
                merged[column] = merged[column].fillna(0.0)
    return merged.sort_values("day").reset_index(drop=True)


def daily_correlation_signals(
    df: pd.DataFrame,
    flow_columns: Iterable[str],
    flow_window_days: int,
) -> pd.DataFrame:
    """Build daily P3A and cargo-change signals for lead/lag correlation.

    P3A is represented by its return between consecutive market observations.
    Each cargo signal is the change in a rolling calendar-day total versus the
    preceding equal-length window. A one-day window therefore compares today's
    cargo flow with yesterday's; a seven-day window compares the latest seven
    days with the prior seven days.
    """
    if flow_window_days < 1:
        raise ValueError("flow_window_days must be at least 1")
    if "day" not in df.columns or "p3a_82" not in df.columns:
        raise KeyError("Daily dataset must contain 'day' and 'p3a_82' columns")

    result = df[["day"]].copy()
    baltic = pd.to_numeric(df["p3a_82"], errors="coerce")
    result["p3a_82_return"] = baltic.dropna().pct_change(fill_method=None).reindex(df.index)

    for column in flow_columns:
        if column not in df.columns:
            raise KeyError(f"Missing required flow column: {column}")
        flow = pd.to_numeric(df[column], errors="coerce")
        if flow.notna().sum() == 0:
            result[f"{column}_change"] = np.nan
            continue
        flow = flow.fillna(0.0)
        rolling_total = flow.rolling(
            window=flow_window_days, min_periods=flow_window_days
        ).sum()
        result[f"{column}_change"] = rolling_total.diff(flow_window_days)

    return result


def weekly_correlation_signals(
    df: pd.DataFrame,
    flow_columns: Iterable[str],
) -> pd.DataFrame:
    """Add week-over-week percentage-change signals for P3A and cargo flows.

    Zero prior-week cargo values do not have a meaningful percentage change, so
    their resulting infinite values are retained as missing rather than treated
    as a large positive signal.
    """
    if "week" not in df.columns or "p3a_82" not in df.columns:
        raise KeyError("Weekly dataset must contain 'week' and 'p3a_82' columns")

    result = df.copy()
    signal_columns = ["p3a_82", *flow_columns]
    for column in signal_columns:
        if column not in result.columns:
            raise KeyError(f"Missing required flow column: {column}")
        values = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}_wow"] = values.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan
        )
    return result


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
