from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.analysis import best_lag_summary, correlation_summary, lead_lag_correlations
from src.charts import lag_heatmap, trend_figure
from src.config import DatabaseSettings, SecretsConfigError, get_database_settings
from src.data_access import (
    DataSourceError,
    create_mysql_engine,
    fetch_australia_shipments,
    fetch_baltic_p3a82,
    fetch_china_arrivals,
    fetch_indonesia_shipments,
)
from src.transform import (
    add_change_columns,
    add_indexed_columns,
    build_monthly_dataset,
    monthly_baltic,
    monthly_volume,
)


BASE_COLUMNS = ["p3a_82", "australia_volume", "indonesia_volume", "china_arrivals"]
FLOW_COLUMNS = ["australia_volume", "indonesia_volume", "china_arrivals"]
LABELS = {
    "p3a_82": "Baltic P3A_82",
    "australia_volume": "Australia coal shipments",
    "indonesia_volume": "Indonesia coal shipments",
    "china_arrivals": "China coal arrivals",
    "p3a_82_index": "Baltic P3A_82 index",
    "australia_volume_index": "Australia shipments index",
    "indonesia_volume_index": "Indonesia shipments index",
    "china_arrivals_index": "China arrivals index",
    "p3a_82_mom": "Baltic P3A_82 MoM",
    "australia_volume_mom": "Australia shipments MoM",
    "indonesia_volume_mom": "Indonesia shipments MoM",
    "china_arrivals_mom": "China arrivals MoM",
    "p3a_82_yoy": "Baltic P3A_82 YoY",
    "australia_volume_yoy": "Australia shipments YoY",
    "indonesia_volume_yoy": "Indonesia shipments YoY",
    "china_arrivals_yoy": "China arrivals YoY",
}


st.set_page_config(
    page_title="Correlation Model",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --accent: #002FA7;
            --ink: #111827;
            --muted: #6B7280;
            --rule: #D1D5DB;
            --surface: #FFFFFF;
            --wash: #F7F7F8;
        }
        .stApp {
            background: var(--surface);
            color: var(--ink);
        }
        h1, h2, h3 {
            letter-spacing: 0;
            font-family: "Helvetica Neue", Arial, sans-serif;
        }
        h1 {
            font-size: 2.1rem !important;
            line-height: 1.1 !important;
            overflow-wrap: anywhere;
            white-space: normal;
        }
        div[data-testid="stMetric"] {
            border-top: 1px solid var(--rule);
            border-bottom: 1px solid var(--rule);
            padding: 0.75rem 0;
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink);
        }
        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--rule);
            background: var(--wash);
        }
        .block-container {
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: none;
        }
        .caption-line {
            border-top: 1px solid var(--rule);
            color: var(--muted);
            font-size: 0.9rem;
            padding-top: 0.65rem;
            overflow-wrap: anywhere;
        }
        div[data-testid="stAlert"] p,
        div[data-testid="stMarkdownContainer"] p {
            overflow-wrap: anywhere;
            white-space: normal;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def engine_for(settings: DatabaseSettings):
    return create_mysql_engine(settings)


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Loading database series...")
def load_monthly_dataset(
    axs_settings: DatabaseSettings,
    baltic_settings: DatabaseSettings,
    start_date: date,
) -> pd.DataFrame:
    axs_engine = engine_for(axs_settings)
    baltic_engine = engine_for(baltic_settings)

    baltic_raw = fetch_baltic_p3a82(
        baltic_engine, start_date, schema=baltic_settings.database
    )
    australia_raw = fetch_australia_shipments(axs_engine, start_date)
    indonesia_raw = fetch_indonesia_shipments(axs_engine, start_date)
    china_raw = fetch_china_arrivals(axs_engine, start_date)

    value_col = "volume"
    baltic = monthly_baltic(baltic_raw, "date", "value")
    australia = monthly_volume(
        australia_raw,
        "date",
        value_col if value_col in australia_raw.columns else None,
        "australia_volume",
    )
    indonesia = monthly_volume(
        indonesia_raw,
        "date",
        value_col if value_col in indonesia_raw.columns else None,
        "indonesia_volume",
    )
    china = monthly_volume(
        china_raw,
        "date",
        value_col if value_col in china_raw.columns else None,
        "china_arrivals",
    )

    monthly = build_monthly_dataset(baltic, australia, indonesia, china)
    monthly = add_indexed_columns(monthly, BASE_COLUMNS)
    monthly = add_change_columns(monthly, BASE_COLUMNS)
    return monthly


def format_corr(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def latest_delta(df: pd.DataFrame, column: str) -> float | None:
    change_col = f"{column}_mom"
    if change_col not in df.columns or df.empty:
        return None
    value = df[change_col].dropna()
    if value.empty:
        return None
    return float(value.iloc[-1] * 100)


def render_metric_row(df: pd.DataFrame, corr_df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    corr = corr_df.set_index("series")
    cols = st.columns(4)

    with cols[0]:
        delta = latest_delta(df, "p3a_82")
        st.metric(
            "Baltic P3A_82",
            f"{latest['p3a_82']:,.0f}",
            None if delta is None else f"{delta:.1f}% MoM",
        )

    for col, series in zip(cols[1:], FLOW_COLUMNS, strict=True):
        with col:
            pearson = corr.loc[series, "pearson"] if series in corr.index else pd.NA
            st.metric(LABELS[series], format_corr(pearson))


def mode_columns(mode: str) -> tuple[list[str], str]:
    if mode == "Absolute monthly values":
        return BASE_COLUMNS, "Monthly value"
    if mode == "Month-over-month change":
        return [f"{column}_mom" for column in BASE_COLUMNS], "MoM change"
    if mode == "Year-over-year change":
        return [f"{column}_yoy" for column in BASE_COLUMNS], "YoY change"
    return [f"{column}_index" for column in BASE_COLUMNS], "Indexed level"


def render_dashboard() -> None:
    apply_style()
    st.title("Correlation Model")
    st.markdown(
        '<div class="caption-line">Baltic P3A_82 vs Australia shipments, Indonesia shipments, and China arrivals.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        years = st.slider("Years", min_value=3, max_value=15, value=10, step=1)
        max_lag = st.slider("Lead/lag window", min_value=3, max_value=24, value=12, step=1)
        min_periods = st.slider(
            "Minimum observations", min_value=3, max_value=24, value=6, step=1
        )
        mode = st.radio(
            "Trend view",
            [
                "Indexed level",
                "Absolute monthly values",
                "Month-over-month change",
                "Year-over-year change",
            ],
        )
        st.caption("Positive lag means Baltic is compared with a later flow value.")

    try:
        axs_settings = get_database_settings(st.secrets, "axs")
        baltic_settings = get_database_settings(st.secrets, "baltic")
    except FileNotFoundError:
        st.error("Missing Streamlit secrets.")
        st.info(
            "Add [axs] and [baltic] secrets in Streamlit Cloud Settings. "
            "Use the example secrets file as the template."
        )
        return
    except (KeyError, SecretsConfigError) as exc:
        st.error(str(exc))
        st.info(
            "Add [axs] and [baltic] secrets in Streamlit Cloud Settings. "
            "Use the example secrets file as the template."
        )
        return

    start_date = (pd.Timestamp.today().normalize() - pd.DateOffset(years=years + 1)).date()

    try:
        monthly = load_monthly_dataset(axs_settings, baltic_settings, start_date)
    except (DataSourceError, SQLAlchemyError, KeyError, ValueError) as exc:
        st.error("Unable to load the analysis dataset.")
        st.info(str(exc))
        return

    if monthly.empty:
        st.warning("No overlapping monthly observations were returned.")
        return

    max_month = monthly["month"].max()
    display_start = max_month - pd.DateOffset(years=years)
    monthly = monthly[monthly["month"] >= display_start].reset_index(drop=True)

    if len(monthly) < min_periods:
        st.warning("Not enough overlapping monthly observations for correlation analysis.")
        st.dataframe(monthly, use_container_width=True)
        return

    corr_df = correlation_summary(monthly, "p3a_82", FLOW_COLUMNS, min_periods)
    lag_df = lead_lag_correlations(monthly, "p3a_82", FLOW_COLUMNS, max_lag, min_periods)
    best_lags = best_lag_summary(lag_df)

    render_metric_row(monthly, corr_df)

    trend_columns, value_title = mode_columns(mode)
    st.plotly_chart(
        trend_figure(
            monthly,
            trend_columns,
            "10-year monthly trend",
            LABELS,
            value_title,
        ),
        use_container_width=True,
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.plotly_chart(lag_heatmap(lag_df, LABELS), use_container_width=True)
    with right:
        st.subheader("Lead/lag summary")

        summary = corr_df.merge(
            best_lags,
            on="series",
            how="left",
            suffixes=("", "_lag"),
        )

        if "observations" not in summary.columns:
            if "observations_x" in summary.columns:
                summary["observations"] = summary["observations_x"]
            elif "observations_lag" in summary.columns:
                summary["observations"] = summary["observations_lag"]
            elif "observations_y" in summary.columns:
                summary["observations"] = summary["observations_y"]

        summary["series"] = summary["series"].map(LABELS).fillna(summary["series"])

        display_columns = [
            "series",
            "pearson",
            "spearman",
            "best_lag_months",
            "best_lag_pearson",
            "observations",
        ]

        for column in display_columns:
            if column not in summary.columns:
                summary[column] = pd.NA

        st.dataframe(
            summary[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Correlation is descriptive evidence only. It should not be read as causation."
        )

    with st.expander("Monthly analysis dataset", expanded=False):
        st.dataframe(monthly, use_container_width=True, hide_index=True)
        csv = monthly.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="correlation_model_monthly_dataset.csv",
            mime="text/csv",
        )

    refreshed = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %Z")
    st.caption(f"Last app refresh: {refreshed}")


if __name__ == "__main__":
    render_dashboard()
