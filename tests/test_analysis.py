import math

import pandas as pd

from src.analysis import (
    best_lag_summary,
    correlation_summary,
    lead_lag_correlations,
)


def test_correlation_summary_returns_pearson_and_spearman():
    rows = pd.DataFrame(
        {
            "p3a_82": [1, 2, 3, 4, 5, 6],
            "australia_volume": [2, 4, 6, 8, 10, 12],
            "indonesia_volume": [12, 10, 8, 6, 4, 2],
        }
    )

    result = correlation_summary(
        rows,
        "p3a_82",
        ["australia_volume", "indonesia_volume"],
        min_periods=3,
    )

    by_series = result.set_index("series")
    assert by_series.loc["australia_volume", "pearson"] == 1.0
    assert by_series.loc["australia_volume", "spearman"] == 1.0
    assert by_series.loc["indonesia_volume", "pearson"] == -1.0
    assert by_series.loc["indonesia_volume", "spearman"] == -1.0
    assert by_series.loc["australia_volume", "observations"] == 6


def test_correlation_summary_returns_nan_for_insufficient_observations():
    rows = pd.DataFrame(
        {
            "p3a_82": [1.0, None, 3.0],
            "china_arrivals": [2.0, 4.0, None],
        }
    )

    result = correlation_summary(
        rows,
        "p3a_82",
        ["china_arrivals"],
        min_periods=2,
    )

    record = result.iloc[0]
    assert record["observations"] == 1
    assert math.isnan(record["pearson"])
    assert math.isnan(record["spearman"])


def test_lead_lag_correlations_covers_full_lag_window():
    rows = pd.DataFrame(
        {
            "p3a_82": list(range(30)),
            "australia_volume": list(range(30)),
        }
    )

    result = lead_lag_correlations(
        rows,
        "p3a_82",
        ["australia_volume"],
        max_lag=12,
        min_periods=6,
    )

    assert result["lag_months"].tolist() == list(range(-12, 13))
    assert set(result["series"]) == {"australia_volume"}


def test_lead_lag_correlations_supports_day_column_names():
    rows = pd.DataFrame(
        {
            "p3a_82_return": list(range(10)),
            "australia_volume_change": list(range(10)),
        }
    )

    result = lead_lag_correlations(
        rows,
        "p3a_82_return",
        ["australia_volume_change"],
        max_lag=2,
        min_periods=3,
        lag_column="lag_days",
    )

    assert result["lag_days"].tolist() == [-2, -1, 0, 1, 2]
    assert "lag_months" not in result.columns


def test_best_lag_summary_picks_largest_absolute_correlation():
    lag_rows = pd.DataFrame(
        {
            "series": ["australia_volume", "australia_volume", "indonesia_volume"],
            "lag_months": [-1, 2, 0],
            "pearson": [0.4, -0.8, 0.5],
            "observations": [10, 10, 10],
        }
    )

    result = best_lag_summary(lag_rows)

    by_series = result.set_index("series")
    assert by_series.loc["australia_volume", "best_lag_months"] == 2
    assert by_series.loc["australia_volume", "best_lag_pearson"] == -0.8
    assert by_series.loc["indonesia_volume", "best_lag_months"] == 0
    assert by_series.loc["indonesia_volume", "best_lag_pearson"] == 0.5


def test_best_lag_summary_supports_day_column_names():
    lag_rows = pd.DataFrame(
        {
            "series": ["australia_volume_change", "australia_volume_change"],
            "lag_days": [-1, 3],
            "pearson": [0.4, -0.8],
            "observations": [100, 98],
        }
    )

    result = best_lag_summary(lag_rows, "lag_days", "best_lag_days")

    record = result.iloc[0]
    assert record["best_lag_days"] == 3
    assert record["best_lag_pearson"] == -0.8
