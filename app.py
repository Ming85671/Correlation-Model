from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.analysis import best_lag_summary, correlation_summary, lead_lag_correlations
from src.charts import lag_heatmap, standardized_trend_figure, trend_figure
from src.config import DatabaseSettings, SecretsConfigError, get_database_settings
from src.data_access import (
    DataSourceError,
    create_mysql_engine,
    fetch_australia_shipments,
    fetch_analysis_date_bounds,
    fetch_baltic_p3a82,
    fetch_china_arrivals,
    fetch_indonesia_shipments,
)
from src.transform import (
    add_change_columns,
    add_indexed_columns,
    build_daily_dataset,
    build_monthly_dataset,
    daily_baltic,
    daily_correlation_signals,
    daily_volume,
    monthly_baltic,
    monthly_volume,
)


BASE_COLUMNS = ["p3a_82", "australia_volume", "indonesia_volume", "china_arrivals"]
FLOW_COLUMNS = ["australia_volume", "indonesia_volume", "china_arrivals"]
DAILY_TARGET_COLUMN = "p3a_82_return"
DAILY_FLOW_COLUMNS = [f"{column}_change" for column in FLOW_COLUMNS]
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


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Checking available history...")
def available_history_bounds(
    axs_settings: DatabaseSettings,
    baltic_settings: DatabaseSettings,
) -> tuple[date, date]:
    """Return the date range shared by all series required by the dashboard."""
    return fetch_analysis_date_bounds(
        engine_for(axs_settings),
        engine_for(baltic_settings),
        baltic_settings.database,
    )


def complete_history_years(start_date: date, end_date: date) -> int:
    """Return the number of full years between two inclusive history bounds."""
    years = end_date.year - start_date.year
    if (end_date.month, end_date.day) < (start_date.month, start_date.day):
        years -= 1
    return max(1, years)


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Loading database series...")
def load_datasets(
    axs_settings: DatabaseSettings,
    baltic_settings: DatabaseSettings,
    start_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the source series and return monthly and calendar-day datasets."""
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

    daily = build_daily_dataset(
        daily_baltic(baltic_raw, "date", "value"),
        daily_volume(
            australia_raw,
            "date",
            value_col if value_col in australia_raw.columns else None,
            "australia_volume",
        ),
        daily_volume(
            indonesia_raw,
            "date",
            value_col if value_col in indonesia_raw.columns else None,
            "indonesia_volume",
        ),
        daily_volume(
            china_raw,
            "date",
            value_col if value_col in china_raw.columns else None,
            "china_arrivals",
        ),
    )
    return monthly, daily


def format_corr(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def latest_percent_change(df: pd.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.empty:
        return None
    value = pd.to_numeric(df[column], errors="coerce").dropna()
    if value.empty:
        return None
    return float(value.iloc[-1] * 100)


def render_metric_row(
    latest_p3a: float | None,
    p3a_delta: float | None,
    corr_df: pd.DataFrame,
    feature_columns: list[str],
    feature_labels: list[str],
) -> None:
    corr = corr_df.set_index("series")
    cols = st.columns(4)

    with cols[0]:
        st.metric(
            "Baltic P3A_82",
            "n/a" if latest_p3a is None else f"{latest_p3a:,.0f}",
            None if p3a_delta is None else f"{p3a_delta:.1f}%",
        )

    for col, series, label in zip(cols[1:], feature_columns, feature_labels, strict=True):
        with col:
            pearson = corr.loc[series, "pearson"] if series in corr.index else pd.NA
            st.metric(label, format_corr(pearson))


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

    try:
        history_start, history_end = available_history_bounds(
            axs_settings, baltic_settings
        )
    except (DataSourceError, SQLAlchemyError, KeyError, ValueError) as exc:
        st.error("Unable to determine the available history.")
        st.info(str(exc))
        return

    max_years = complete_history_years(history_start, history_end)
    default_years = min(10, max_years)

    with st.sidebar:
        st.header("Controls")
        frequency = st.radio("Analysis frequency", ["Monthly", "Daily"])
        years = st.number_input(
            "Years", min_value=1, max_value=max_years, value=default_years, step=1
        )
        st.caption(
            f"Shared data history: {history_start:%Y-%m-%d} to {history_end:%Y-%m-%d}."
        )
        if frequency == "Daily":
            flow_window_days = st.number_input(
                "Cargo change window (days)", min_value=1, max_value=30, value=7, step=1
            )
            max_lag = st.number_input(
                "Lead/lag window (days)", min_value=0, max_value=60, value=30, step=1
            )
            min_periods = st.number_input(
                "Minimum observations", min_value=60, max_value=500, value=120, step=1
            )
            mode = None
            st.caption(
                "Daily mode compares P3A returns with changes in rolling cargo totals."
            )
        else:
            flow_window_days = None
            max_lag = st.number_input(
                "Lead/lag window (months)", min_value=0, max_value=24, value=12, step=1
            )
            min_periods = st.number_input(
                "Minimum observations", min_value=6, max_value=60, value=12, step=1
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
        st.caption(
            "Positive lag means cargo changes occur first and P3A is compared later."
        )

    start_date = (
        pd.Timestamp(history_end) - pd.DateOffset(years=int(years) + 1)
    ).date()

    try:
        monthly, daily = load_datasets(axs_settings, baltic_settings, start_date)
    except (DataSourceError, SQLAlchemyError, KeyError, ValueError) as exc:
        st.error("Unable to load the analysis datasets.")
        st.info(str(exc))
        return

    if frequency == "Daily":
        market_days = daily.dropna(subset=["p3a_82"])
        if market_days.empty:
            st.warning("No daily Baltic observations were returned.")
            return

        display_start = market_days["day"].max() - pd.DateOffset(years=years)
        source_data = daily[daily["day"] >= display_start].reset_index(drop=True)
        daily_signals = daily_correlation_signals(daily, FLOW_COLUMNS, int(flow_window_days))
        analysis_data = daily_signals[
            daily_signals["day"] >= display_start
        ].reset_index(drop=True)
        target_column = DAILY_TARGET_COLUMN
        feature_columns = DAILY_FLOW_COLUMNS
        series_labels = {
            DAILY_TARGET_COLUMN: "P3A daily return",
            "australia_volume_change": f"Australia {flow_window_days}-day flow change",
            "indonesia_volume_change": f"Indonesia {flow_window_days}-day flow change",
            "china_arrivals_change": f"China {flow_window_days}-day flow change",
        }
        lag_column = "lag_days"
        best_lag_column = "best_lag_days"
        lag_title = "Lag days"
        latest_p3a_values = source_data["p3a_82"].dropna()
        latest_p3a = (
            float(latest_p3a_values.iloc[-1]) if not latest_p3a_values.empty else None
        )
        p3a_delta = latest_percent_change(analysis_data, DAILY_TARGET_COLUMN)
        dataset_title = "Daily correlation signals"
        dataset_filename = "correlation_model_daily_signals.csv"
    else:
        if monthly.empty:
            st.warning("No overlapping monthly observations were returned.")
            return

        display_start = monthly["month"].max() - pd.DateOffset(years=years)
        source_data = monthly[monthly["month"] >= display_start].reset_index(drop=True)
        analysis_data = source_data
        target_column = "p3a_82"
        feature_columns = FLOW_COLUMNS
        series_labels = LABELS
        lag_column = "lag_months"
        best_lag_column = "best_lag_months"
        lag_title = "Lag months"
        latest_p3a_values = source_data["p3a_82"].dropna()
        latest_p3a = (
            float(latest_p3a_values.iloc[-1]) if not latest_p3a_values.empty else None
        )
        p3a_delta = latest_percent_change(source_data, "p3a_82_mom")
        dataset_title = "Monthly analysis dataset"
        dataset_filename = "correlation_model_monthly_dataset.csv"

    if len(analysis_data) < min_periods:
        st.warning("Not enough observations for correlation analysis.")
        st.dataframe(analysis_data, use_container_width=True)
        return

    corr_df = correlation_summary(
        analysis_data, target_column, feature_columns, int(min_periods)
    )
    lag_df = lead_lag_correlations(
        analysis_data,
        target_column,
        feature_columns,
        int(max_lag),
        int(min_periods),
        lag_column,
    )
    best_lags = best_lag_summary(lag_df, lag_column, best_lag_column)

    render_metric_row(
        latest_p3a,
        p3a_delta,
        corr_df,
        feature_columns,
        [LABELS[column] for column in FLOW_COLUMNS],
    )

    if frequency == "Monthly":
        trend_columns, value_title = mode_columns(mode)
        st.plotly_chart(
            trend_figure(
                source_data,
                trend_columns,
                f"{years}-year monthly trend",
                LABELS,
                value_title,
            ),
            use_container_width=True,
        )
    else:
        st.subheader("Daily movement relationship")
        st.caption(
            f"P3A daily return is correlated with the change in each {flow_window_days}-day "
            "cargo total. This removes the misleading effect of comparing two long-term "
            "levels that simply trend over time."
        )
        st.plotly_chart(
            standardized_trend_figure(
                analysis_data,
                [target_column, *feature_columns],
                "Daily trend: P3A and cargo-movement signals",
                series_labels,
            ),
            use_container_width=True,
        )
        st.caption(
            "Each line is standardized around its own average: 0 is typical, while +1 and -1 "
            "are one standard deviation above and below that series' average."
        )

    left, right = st.columns([1.15, 0.85])
    with left:
        st.plotly_chart(
            lag_heatmap(lag_df, series_labels, lag_column, lag_title),
            use_container_width=True,
        )
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

        summary["series"] = summary["series"].map(series_labels).fillna(summary["series"])

        display_columns = [
            "series",
            "pearson",
            "spearman",
            best_lag_column,
            "best_lag_pearson",
            "observations",
        ]

        for column in display_columns:
            if column not in summary.columns:
                summary[column] = pd.NA

        summary_display = summary[display_columns].rename(
            columns={best_lag_column: f"Best lag ({lag_title.lower().replace('lag ', '')})"}
        )
        st.dataframe(
            summary_display,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Correlation is descriptive evidence only. It should not be read as causation."
        )

    with st.expander(dataset_title, expanded=False):
        st.dataframe(analysis_data, use_container_width=True, hide_index=True)
        csv = analysis_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=dataset_filename,
            mime="text/csv",
        )

    refreshed = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S %Z")
    st.caption(f"Last app refresh: {refreshed}")


if __name__ == "__main__":
    render_dashboard()
