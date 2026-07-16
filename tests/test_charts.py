import pandas as pd

from src.charts import standardized_trend_figure


def test_standardized_trend_figure_aligns_daily_series_on_a_common_scale():
    rows = pd.DataFrame(
        {
            "day": pd.date_range("2026-01-01", periods=3, freq="D"),
            "p3a_82_return": [1.0, 2.0, 3.0],
            "australia_volume_change": [10.0, 20.0, 30.0],
        }
    )

    figure = standardized_trend_figure(
        rows,
        ["p3a_82_return", "australia_volume_change"],
        "Daily trend",
        {
            "p3a_82_return": "P3A daily return",
            "australia_volume_change": "Australia flow change",
        },
        connect_gaps_columns=["p3a_82_return"],
    )

    assert figure.layout.title.text == "Daily trend"
    assert figure.layout.yaxis.title.text == "Standard deviations from each series' average"
    assert [trace.name for trace in figure.data] == [
        "P3A daily return",
        "Australia flow change",
    ]
    assert round(float(figure.data[0].y[1]), 6) == 0.0
    assert round(float(figure.data[1].y[1]), 6) == 0.0
    assert figure.data[0].connectgaps is True
    assert figure.data[1].connectgaps is None
