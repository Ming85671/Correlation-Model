from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def trend_figure(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    labels: dict[str, str] | None = None,
    value_title: str = "Value",
    time_column: str = "month",
    time_label: str = "Month",
) -> go.Figure:
    labels = labels or {}
    plot_df = df[[time_column, *columns]].copy()
    long_df = plot_df.melt(time_column, var_name="series", value_name="value")
    long_df["series"] = long_df["series"].map(lambda value: labels.get(value, value))

    fig = px.line(
        long_df,
        x=time_column,
        y="value",
        color="series",
        title=title,
        labels={time_column: time_label, "value": value_title, "series": "Series"},
    )
    fig.update_layout(
        hovermode="x unified",
        legend_title_text="",
        margin=dict(l=8, r=8, t=54, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Helvetica Neue, Arial, sans-serif", color="#111827"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E7EB", zeroline=False)
    return fig


def lag_heatmap(
    lag_df: pd.DataFrame,
    labels: dict[str, str] | None = None,
    lag_column: str = "lag_months",
    lag_title: str = "Lag months",
) -> go.Figure:
    """Render a lead/lag Pearson-correlation heatmap."""
    labels = labels or {}
    if lag_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Lead/lag correlation heatmap")
        return fig

    rows = lag_df.copy()
    rows["series_label"] = rows["series"].map(lambda value: labels.get(value, value))
    pivot = rows.pivot(index="series_label", columns=lag_column, values="pearson")

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        labels=dict(x=lag_title, y="Series", color="Pearson"),
        title="Lead/lag correlation heatmap",
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=54, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Helvetica Neue, Arial, sans-serif", color="#111827"),
    )
    return fig
