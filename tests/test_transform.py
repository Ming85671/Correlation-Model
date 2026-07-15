import numpy as np
import pandas as pd

from src.transform import (
    add_change_columns,
    add_indexed_columns,
    build_daily_dataset,
    build_monthly_dataset,
    daily_baltic,
    daily_correlation_signals,
    daily_flow_metrics,
    daily_volume,
    monthly_baltic,
    monthly_flow_metrics,
    monthly_volume,
)


def test_monthly_volume_groups_by_month_and_sums_values():
    rows = pd.DataFrame(
        {
            "load_start_date": ["2026-01-03", "2026-01-18", "2026-02-02"],
            "volume": [10_000, 15_000, 20_000],
        }
    )

    result = monthly_volume(rows, "load_start_date", "volume", "australia_volume")

    assert result.to_dict("records") == [
        {"month": pd.Timestamp("2026-01-01"), "australia_volume": 25_000},
        {"month": pd.Timestamp("2026-02-01"), "australia_volume": 20_000},
    ]


def test_daily_volume_groups_by_calendar_day_and_sums_values():
    rows = pd.DataFrame(
        {
            "load_start_date": ["2026-01-03 08:00", "2026-01-03 18:00", "2026-01-04"],
            "volume": [10_000, 15_000, 20_000],
        }
    )

    result = daily_volume(rows, "load_start_date", "volume", "australia_volume")

    assert result.to_dict("records") == [
        {"day": pd.Timestamp("2026-01-03"), "australia_volume": 25_000},
        {"day": pd.Timestamp("2026-01-04"), "australia_volume": 20_000},
    ]


def test_daily_flow_metrics_keeps_shipment_count_and_volume_separate():
    rows = pd.DataFrame(
        {
            "load_start_date": ["2026-01-03", "2026-01-03", "2026-01-04"],
            "volume": [10_000, 15_000, 20_000],
        }
    )

    result = daily_flow_metrics(
        rows,
        "load_start_date",
        "volume",
        "australia_shipment_count",
        "australia_volume",
    )

    assert result.to_dict("records") == [
        {
            "day": pd.Timestamp("2026-01-03"),
            "australia_shipment_count": 2,
            "australia_volume": 25_000,
        },
        {
            "day": pd.Timestamp("2026-01-04"),
            "australia_shipment_count": 1,
            "australia_volume": 20_000,
        },
    ]


def test_monthly_volume_counts_rows_when_value_column_missing():
    rows = pd.DataFrame(
        {
            "discharge_start_date": ["2026-01-03", "2026-01-18", "2026-02-02"],
        }
    )

    result = monthly_volume(rows, "discharge_start_date", None, "china_arrivals")

    assert result.to_dict("records") == [
        {"month": pd.Timestamp("2026-01-01"), "china_arrivals": 2},
        {"month": pd.Timestamp("2026-02-01"), "china_arrivals": 1},
    ]


def test_monthly_flow_metrics_marks_volume_unavailable_without_reusing_counts():
    rows = pd.DataFrame(
        {"discharge_start_date": ["2026-01-03", "2026-01-18", "2026-02-02"]}
    )

    result = monthly_flow_metrics(
        rows,
        "discharge_start_date",
        None,
        "china_arrival_count",
        "china_arrivals_volume",
    )

    assert result["china_arrival_count"].tolist() == [2, 1]
    assert result["china_arrivals_volume"].isna().all()


def test_monthly_baltic_averages_values_by_month():
    rows = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-02-01"],
            "value": [1000.0, 1100.0, 900.0],
        }
    )

    result = monthly_baltic(rows, "date", "value")

    assert result.to_dict("records") == [
        {"month": pd.Timestamp("2026-01-01"), "p3a_82": 1050.0},
        {"month": pd.Timestamp("2026-02-01"), "p3a_82": 900.0},
    ]


def test_daily_baltic_averages_duplicate_observations_per_day():
    rows = pd.DataFrame(
        {
            "date": ["2026-01-01 08:00", "2026-01-01 18:00", "2026-01-02"],
            "value": [1000.0, 1100.0, 900.0],
        }
    )

    result = daily_baltic(rows, "date", "value")

    assert result.to_dict("records") == [
        {"day": pd.Timestamp("2026-01-01"), "p3a_82": 1050.0},
        {"day": pd.Timestamp("2026-01-02"), "p3a_82": 900.0},
    ]


def test_build_monthly_dataset_keeps_overlapping_months_only():
    baltic = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"]),
            "p3a_82": [100.0, 110.0, 120.0],
        }
    )
    australia = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "australia_volume": [10.0, 12.0],
        }
    )
    indonesia = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-02-01", "2026-03-01"]),
            "indonesia_volume": [20.0, 22.0],
        }
    )
    china = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-02-01", "2026-03-01"]),
            "china_arrivals": [30.0, 32.0],
        }
    )

    result = build_monthly_dataset(baltic, australia, indonesia, china)

    assert result.to_dict("records") == [
        {
            "month": pd.Timestamp("2026-02-01"),
            "p3a_82": 110.0,
            "australia_volume": 12.0,
            "indonesia_volume": 20.0,
            "china_arrivals": 30.0,
        }
    ]


def test_build_daily_dataset_keeps_zero_cargo_days_and_missing_baltic_days():
    baltic = pd.DataFrame(
        {
            "day": pd.to_datetime(["2026-01-02", "2026-01-05"]),
            "p3a_82": [100.0, 110.0],
        }
    )
    australia = pd.DataFrame(
        {"day": pd.to_datetime(["2026-01-03"]), "australia_volume": [20.0]}
    )
    indonesia = pd.DataFrame(columns=["day", "indonesia_volume"])
    china = pd.DataFrame(
        {"day": pd.to_datetime(["2026-01-04"]), "china_arrivals": [30.0]}
    )

    result = build_daily_dataset(baltic, australia, indonesia, china)

    assert result["day"].tolist() == list(pd.date_range("2026-01-02", "2026-01-05"))
    assert result["australia_volume"].tolist() == [0.0, 20.0, 0.0, 0.0]
    assert result["china_arrivals"].tolist() == [0.0, 0.0, 30.0, 0.0]
    assert pd.isna(result.loc[1, "p3a_82"])


def test_daily_correlation_signals_uses_market_returns_and_rolling_flow_change():
    rows = pd.DataFrame(
        {
            "day": pd.date_range("2026-01-01", periods=5, freq="D"),
            "p3a_82": [100.0, 110.0, None, 121.0, 133.1],
            "australia_volume": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    result = daily_correlation_signals(rows, ["australia_volume"], flow_window_days=2)

    assert pd.isna(result.loc[0, "p3a_82_return"])
    assert pd.isna(result.loc[2, "p3a_82_return"])
    assert result["p3a_82_return"].dropna().round(4).tolist() == [0.1, 0.1, 0.1]
    assert result["australia_volume_change"].dropna().tolist() == [40.0, 40.0]


def test_add_indexed_columns_starts_each_series_at_100():
    rows = pd.DataFrame({"p3a_82": [50.0, 75.0], "australia_volume": [20.0, 10.0]})

    result = add_indexed_columns(rows, ["p3a_82", "australia_volume"])

    assert result["p3a_82_index"].tolist() == [100.0, 150.0]
    assert result["australia_volume_index"].tolist() == [100.0, 50.0]


def test_add_indexed_columns_handles_zero_base_as_nan():
    rows = pd.DataFrame({"p3a_82": [0.0, 75.0]})

    result = add_indexed_columns(rows, ["p3a_82"])

    assert np.isnan(result.loc[0, "p3a_82_index"])
    assert np.isnan(result.loc[1, "p3a_82_index"])


def test_add_change_columns_calculates_mom_and_yoy():
    rows = pd.DataFrame(
        {
            "p3a_82": [
                100.0,
                110.0,
                121.0,
                130.0,
                140.0,
                150.0,
                160.0,
                170.0,
                180.0,
                190.0,
                200.0,
                210.0,
                220.0,
            ]
        }
    )

    result = add_change_columns(rows, ["p3a_82"])

    assert round(result.loc[1, "p3a_82_mom"], 4) == 0.1
    assert round(result.loc[12, "p3a_82_yoy"], 4) == 1.2
