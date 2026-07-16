from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from src.analysis import (
    correlation_summary,
    lead_lag_correlations,
    recommended_min_observations,
    top_lag_relationships,
)
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
    monthly_baltic,
    daily_flow_metrics,
    monthly_flow_metrics,
)


VOLUME_FLOW_COLUMNS = [
    "australia_volume",
    "indonesia_volume",
    "china_arrivals_volume",
]
SHIPMENT_COUNT_COLUMNS = [
    "australia_shipment_count",
    "indonesia_shipment_count",
    "china_arrival_count",
]
BASE_COLUMNS = ["p3a_82", *VOLUME_FLOW_COLUMNS, *SHIPMENT_COUNT_COLUMNS]
CARGO_MEASURES = {
    "Shipment count": SHIPMENT_COUNT_COLUMNS,
    "Cargo volume": VOLUME_FLOW_COLUMNS,
}
DATASET_CACHE_VERSION = "cargo-volume-schema-status-v6"
LABELS = {
    "p3a_82": "Baltic P3A_82",
    "australia_shipment_count": "Australia shipment count",
    "indonesia_shipment_count": "Indonesia shipment count",
    "china_arrival_count": "China arrival count",
    "australia_volume": "Australia cargo volume",
    "indonesia_volume": "Indonesia cargo volume",
    "china_arrivals_volume": "China arrival volume",
    "p3a_82_index": "Baltic P3A_82 index",
    "australia_shipment_count_index": "Australia shipment count index",
    "indonesia_shipment_count_index": "Indonesia shipment count index",
    "china_arrival_count_index": "China arrival count index",
    "australia_volume_index": "Australia cargo volume index",
    "indonesia_volume_index": "Indonesia cargo volume index",
    "china_arrivals_volume_index": "China arrival volume index",
    "p3a_82_mom": "Baltic P3A_82 MoM",
    "australia_shipment_count_mom": "Australia shipment count MoM",
    "indonesia_shipment_count_mom": "Indonesia shipment count MoM",
    "china_arrival_count_mom": "China arrival count MoM",
    "australia_volume_mom": "Australia cargo volume MoM",
    "indonesia_volume_mom": "Indonesia cargo volume MoM",
    "china_arrivals_volume_mom": "China arrival volume MoM",
    "p3a_82_yoy": "Baltic P3A_82 YoY",
    "australia_shipment_count_yoy": "Australia shipment count YoY",
    "indonesia_shipment_count_yoy": "Indonesia shipment count YoY",
    "china_arrival_count_yoy": "China arrival count YoY",
    "australia_volume_yoy": "Australia cargo volume YoY",
    "indonesia_volume_yoy": "Indonesia cargo volume YoY",
    "china_arrivals_volume_yoy": "China arrival volume YoY",
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


def max_selectable_months(start_date: date, end_date: date) -> int:
    """Keep the existing full-year history limit while exposing monthly choices."""
    return complete_history_years(start_date, end_date) * 12


@st.cache_data(ttl=60 * 60 * 6, show_spinner="Loading database series...")
def load_datasets(
    axs_settings: DatabaseSettings,
    baltic_settings: DatabaseSettings,
    start_date: date,
    cache_version: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
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
    australia = monthly_flow_metrics(
        australia_raw,
        "date",
        value_col if value_col in australia_raw.columns else None,
        "australia_shipment_count",
        "australia_volume",
    )
    indonesia = monthly_flow_metrics(
        indonesia_raw,
        "date",
        value_col if value_col in indonesia_raw.columns else None,
        "indonesia_shipment_count",
        "indonesia_volume",
    )
    china = monthly_flow_metrics(
        china_raw,
        "date",
        value_col if value_col in china_raw.columns else None,
        "china_arrival_count",
        "china_arrivals_volume",
    )

    monthly = build_monthly_dataset(baltic, australia, indonesia, china)
    monthly = add_indexed_columns(monthly, BASE_COLUMNS)
    monthly = add_change_columns(monthly, BASE_COLUMNS)

    daily = build_daily_dataset(
        daily_baltic(baltic_raw, "date", "value"),
        daily_flow_metrics(
            australia_raw,
            "date",
            value_col if value_col in australia_raw.columns else None,
            "australia_shipment_count",
            "australia_volume",
        ),
        daily_flow_metrics(
            indonesia_raw,
            "date",
            value_col if value_col in indonesia_raw.columns else None,
            "indonesia_shipment_count",
            "indonesia_volume",
        ),
        daily_flow_metrics(
            china_raw,
            "date",
            value_col if value_col in china_raw.columns else None,
            "china_arrival_count",
            "china_arrivals_volume",
        ),
    )
    volume_status = {
        "Australia": {
            "candidate_stats": australia_raw.attrs.get("volume_candidate_stats", []),
            "related_columns": australia_raw.attrs.get("volume_related_columns", []),
        },
        "Indonesia": {
            "candidate_stats": indonesia_raw.attrs.get("volume_candidate_stats", []),
            "related_columns": indonesia_raw.attrs.get("volume_related_columns", []),
        },
        "China": {
            "candidate_stats": china_raw.attrs.get("volume_candidate_stats", []),
            "related_columns": china_raw.attrs.get("volume_related_columns", []),
        },
    }
    return monthly, daily, volume_status


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


def latest_level_change(df: pd.DataFrame, column: str) -> float | None:
    """Return the latest consecutive-observation percentage change of a level."""
    if column not in df.columns or df.empty:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if len(values) < 2:
        return None
    changes = values.pct_change(fill_method=None).dropna()
    return None if changes.empty else float(changes.iloc[-1] * 100)


def describe_lag(lag: int, unit: str) -> str:
    """Explain the direction of a feature lag relative to P3A."""
    if lag == 0:
        return "Same period"
    if lag > 0:
        return f"Cargo leads P3A by {lag} {unit}"
    return f"P3A leads cargo by {abs(lag)} {unit}"


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
            st.metric(f"{label} Pearson", format_corr(pearson))


def mode_columns(mode: str, flow_columns: list[str]) -> tuple[list[str], str]:
    columns = ["p3a_82", *flow_columns]
    if mode == "Absolute monthly values":
        return columns, "Monthly value"
    if mode == "Month-over-month change":
        return [f"{column}_mom" for column in columns], "MoM change"
    if mode == "Year-over-year change":
        return [f"{column}_yoy" for column in columns], "YoY change"
    return [f"{column}_index" for column in columns], "Indexed level"


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

    max_months = max_selectable_months(history_start, history_end)
    default_months = min(10 * 12, max_months)

    with st.sidebar:
        st.header("Controls")
        frequency = st.radio("Analysis frequency", ["Monthly", "Daily"])
        cargo_measure = st.radio("Cargo measure", list(CARGO_MEASURES))
        months = int(
            st.number_input(
                "Months",
                min_value=1,
                max_value=max_months,
                value=default_months,
                step=1,
            )
        )
        st.caption(
            f"Shared data history: {history_start:%Y-%m-%d} to {history_end:%Y-%m-%d}."
        )
        if frequency == "Daily":
            mode = None
        else:
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
            "Lead/lag relationships and their reliability threshold are calculated automatically. "
            "A positive lag means cargo occurs first and P3A is compared later."
        )

    # Load one additional year so year-over-year changes remain available.
    start_date = (
        pd.Timestamp(history_end) - pd.DateOffset(months=months + 12)
    ).date()

    try:
        monthly, daily, volume_status = load_datasets(
            axs_settings,
            baltic_settings,
            start_date,
            DATASET_CACHE_VERSION,
        )
    except (DataSourceError, SQLAlchemyError, KeyError, ValueError) as exc:
        st.error("Unable to load the analysis datasets.")
        st.info(str(exc))
        return

    if frequency == "Daily":
        market_days = daily.dropna(subset=["p3a_82"])
        if market_days.empty:
            st.warning("No daily Baltic observations were returned.")
            return

        display_start = market_days["day"].max() - pd.DateOffset(months=months)
        source_data = daily[daily["day"] >= display_start].reset_index(drop=True)
        active_flow_columns = CARGO_MEASURES[cargo_measure]
        analysis_data = source_data
        target_column = "p3a_82"
        feature_columns = active_flow_columns
        series_labels = {"p3a_82": LABELS["p3a_82"]}
        series_labels.update({column: LABELS[column] for column in active_flow_columns})
        lag_column = "lag_days"
        lag_title = "Lag days"
        max_lag = 60
        latest_p3a_values = source_data["p3a_82"].dropna()
        latest_p3a = (
            float(latest_p3a_values.iloc[-1]) if not latest_p3a_values.empty else None
        )
        p3a_delta = latest_level_change(source_data, "p3a_82")
        dataset_title = "Daily analysis dataset"
        dataset_filename = "correlation_model_daily_dataset.csv"
    else:
        if monthly.empty:
            st.warning("No overlapping monthly observations were returned.")
            return

        display_start = monthly["month"].max() - pd.DateOffset(months=months)
        source_data = monthly[monthly["month"] >= display_start].reset_index(drop=True)
        analysis_data = source_data
        target_column = "p3a_82"
        active_flow_columns = CARGO_MEASURES[cargo_measure]
        feature_columns = active_flow_columns
        series_labels = {"p3a_82": LABELS["p3a_82"]}
        series_labels.update(
            {column: LABELS[column] for column in active_flow_columns}
        )
        lag_column = "lag_months"
        lag_title = "Lag months"
        max_lag = 24
        latest_p3a_values = source_data["p3a_82"].dropna()
        latest_p3a = (
            float(latest_p3a_values.iloc[-1]) if not latest_p3a_values.empty else None
        )
        p3a_delta = latest_percent_change(source_data, "p3a_82_mom")
        dataset_title = "Monthly analysis dataset"
        dataset_filename = "correlation_model_monthly_dataset.csv"

    if analysis_data.empty:
        st.warning("Not enough observations for correlation analysis.")
        st.dataframe(analysis_data, use_container_width=True)
        return

    min_periods = recommended_min_observations(
        analysis_data, target_column, feature_columns, frequency
    )

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
    top_lags = top_lag_relationships(lag_df, lag_column, limit=10)

    render_metric_row(
        latest_p3a,
        p3a_delta,
        corr_df,
        feature_columns,
        [LABELS[column] for column in active_flow_columns],
    )
    st.caption(
        "Cargo cards show the same-period Pearson correlation with P3A; they do not show cargo totals."
    )
    if cargo_measure == "Cargo volume":
        unavailable = [
            column
            for column in active_flow_columns
            if source_data[column].notna().sum() == 0
        ]
        if unavailable:
            st.warning(
                "The AXS source has no usable cargo-volume values for the selected flow. "
                "See the data-status panel for the source-field check."
            )
            with st.expander("Cargo-volume source status"):
                for flow, status in volume_status.items():
                    st.caption(flow)
                    candidate_stats = status["candidate_stats"]
                    if candidate_stats:
                        st.dataframe(pd.DataFrame(candidate_stats), hide_index=True)
                    else:
                        related_columns = status["related_columns"]
                        if related_columns:
                            st.code(", ".join(related_columns), language=None)
                        else:
                            st.caption("No cargo- or capacity-related source fields found.")

    if frequency == "Monthly":
        trend_columns, value_title = mode_columns(mode, active_flow_columns)
        st.plotly_chart(
            trend_figure(
                source_data,
                trend_columns,
                f"Monthly trend — last {months} months",
                LABELS,
                value_title,
            ),
            use_container_width=True,
        )
    else:
        st.subheader("Daily trend")
        st.caption(
            f"P3A levels and daily {cargo_measure.lower()} are shown on a standardized scale."
        )
        st.plotly_chart(
            standardized_trend_figure(
                source_data,
                [target_column, *feature_columns],
                "Daily trend: P3A and cargo levels",
                series_labels,
                connect_gaps_columns=[target_column],
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
        st.subheader("Top 10 lead/lag relationships")
        if top_lags.empty:
            st.warning("No lead/lag relationship has enough paired observations.")
        else:
            lag_unit = "days" if frequency == "Daily" else "months"
            top_lags = top_lags.copy()
            top_lags["series"] = top_lags["series"].map(series_labels).fillna(
                top_lags["series"]
            )
            top_lags["relationship"] = top_lags[lag_column].map(
                lambda lag: describe_lag(int(lag), lag_unit)
            )
            st.dataframe(
                top_lags[["series", "relationship", "pearson", "observations"]],
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            f"Automatic reliability threshold: {min_periods} paired observations. "
            "The top 10 are ranked by absolute Pearson correlation; the sign still shows "
            "whether the relationship moves together or in opposite directions. Correlation "
            "is descriptive evidence only, not causation."
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
