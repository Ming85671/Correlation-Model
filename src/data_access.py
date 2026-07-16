from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from src.config import DatabaseSettings


class DataSourceError(RuntimeError):
    """Raised when a required database source cannot be found or queried."""


@dataclass(frozen=True)
class BalticSource:
    schema: str
    table: str
    date_column: str
    value_column: str


AXS_SCHEMA = "axs"
AXS_TABLE = "axs"
AXS_VOLUME_CANDIDATES = (
    "cargo_quantity",
    "cargo_qty",
    "cargo_volume",
    "cargo_tonnage",
    "cargo_tonnes",
    "cargo_tons",
    "cargo_metric_tons",
    "cargo_mt",
    "cargo_weight",
    "quantity",
    "volume",
    "mt",
    "metric_tons",
    "tonnes",
    "tons",
)
BALTIC_DATE_CANDIDATES = (
    "date",
    "pricedate",
    "price_date",
    "assessment_date",
    "observation_date",
    "report_date",
    "timestamp",
)
BALTIC_VALUE_CANDIDATES = ("value", "price", "index", "close", "settle", "p3a_82")


def create_mysql_engine(settings: DatabaseSettings) -> Engine:
    user = quote_plus(settings.user)
    password = quote_plus(settings.password)
    host = settings.host
    database = settings.database
    url = f"mysql+pymysql://{user}:{password}@{host}:3306/{database}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"


def _start_date_value(start_date: date | datetime | str) -> date:
    return pd.Timestamp(start_date).date()


def _table_columns(engine: Engine, schema: str, table: str) -> list[str]:
    query = text(
        """
        SELECT COLUMN_NAME
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = :schema
          AND TABLE_NAME = :table
        ORDER BY ORDINAL_POSITION
        """
    )
    rows = pd.read_sql(query, engine, params={"schema": schema, "table": table})
    return [str(value) for value in rows["COLUMN_NAME"].tolist()]


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    by_normalized = {_normalize(column): column for column in columns}
    for candidate in candidates:
        match = by_normalized.get(_normalize(candidate))
        if match:
            return match
    return None


def _axs_volume_columns(columns: Iterable[str]) -> list[str]:
    """Return all cargo-volume candidates, including source-specific unit suffixes."""
    column_list = list(columns)
    by_normalized = {_normalize(column): column for column in column_list}
    matches: list[str] = []
    for candidate in AXS_VOLUME_CANDIDATES:
        match = by_normalized.get(_normalize(candidate))
        if match and match not in matches:
            matches.append(match)

    volume_tokens = (
        "quantity",
        "qty",
        "volume",
        "tonnage",
        "tonne",
        "ton",
        "metricton",
        "mt",
        "weight",
    )
    for column in column_list:
        normalized = _normalize(column)
        if (
            "cargo" in normalized
            and any(token in normalized for token in volume_tokens)
            and column not in matches
        ):
            matches.append(column)
    return matches


def _find_axs_volume_column(columns: Iterable[str]) -> str | None:
    """Find the first cargo-volume field for callers that need one candidate."""
    matches = _axs_volume_columns(columns)
    return matches[0] if matches else None


def _numeric_volume_values(values: pd.Series) -> pd.Series:
    normalized = (
        values.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(r"(?i)\s*(?:mt|metric tons?|tonnes?|tons?)\s*$", "", regex=True)
    )
    return pd.to_numeric(normalized, errors="coerce")


def _best_volume_column(rows: pd.DataFrame, candidates: list[str]) -> str | None:
    """Select the candidate with the most usable, non-constant cargo observations."""
    best_column: str | None = None
    best_score = (0, 0, 0)
    for position, column in enumerate(candidates):
        values = _numeric_volume_values(rows[column])
        usable = values.dropna()
        score = (len(usable), usable.nunique(), -position)
        if usable.nunique() > 1 and score > best_score:
            best_column = column
            best_score = score
    return best_column


def _fetch_axs_rows(
    engine: Engine,
    start_date: date | datetime | str,
    date_col: str,
    filters: str,
    params: dict[str, object] | None = None,
) -> pd.DataFrame:
    columns = _table_columns(engine, AXS_SCHEMA, AXS_TABLE)
    volume_columns = _axs_volume_columns(columns)
    selected = [f"{_quote_identifier(date_col)} AS date"]
    for index, volume_column in enumerate(volume_columns):
        selected.append(f"{_quote_identifier(volume_column)} AS volume_{index}")

    query = text(
        f"""
        SELECT {", ".join(selected)}
        FROM {_quote_identifier(AXS_SCHEMA)}.{_quote_identifier(AXS_TABLE)}
        WHERE {filters}
          AND {_quote_identifier(date_col)} >= :start_date
        ORDER BY {_quote_identifier(date_col)}
        """
    )
    query_params: dict[str, object] = {"start_date": _start_date_value(start_date)}
    if params:
        query_params.update(params)
    rows = pd.read_sql(query, engine, params=query_params)
    if not volume_columns:
        return rows

    aliases = [f"volume_{index}" for index in range(len(volume_columns))]
    selected_alias = _best_volume_column(rows, aliases)
    if selected_alias is None:
        return rows[["date"]]

    return rows[["date", selected_alias]].rename(columns={selected_alias: "volume"})


def _fetch_axs_date_bounds(
    engine: Engine,
    date_col: str,
    filters: str,
    params: dict[str, object],
) -> tuple[date, date]:
    """Return the earliest and latest dates for one filtered AXS series."""
    query = text(
        f"""
        SELECT
          MIN({_quote_identifier(date_col)}) AS earliest_date,
          MAX({_quote_identifier(date_col)}) AS latest_date
        FROM {_quote_identifier(AXS_SCHEMA)}.{_quote_identifier(AXS_TABLE)}
        WHERE {filters}
        """
    )
    rows = pd.read_sql(query, engine, params=params)
    return _read_date_bounds(rows, "AXS series")


def _read_date_bounds(rows: pd.DataFrame, source_name: str) -> tuple[date, date]:
    if rows.empty:
        raise DataSourceError(f"No date bounds were returned for {source_name}.")

    earliest = pd.to_datetime(rows.loc[0, "earliest_date"], errors="coerce")
    latest = pd.to_datetime(rows.loc[0, "latest_date"], errors="coerce")
    if pd.isna(earliest) or pd.isna(latest):
        raise DataSourceError(f"No dated observations were found for {source_name}.")
    return earliest.date(), latest.date()


def _common_date_bounds(bounds: Iterable[tuple[date, date]]) -> tuple[date, date]:
    values = list(bounds)
    if not values:
        raise DataSourceError("No source date bounds were provided.")

    earliest = max(start for start, _ in values)
    latest = min(end for _, end in values)
    if earliest > latest:
        raise DataSourceError("The required series do not have an overlapping date range.")
    return earliest, latest


def fetch_australia_shipments(
    engine: Engine, start_date: date | datetime | str
) -> pd.DataFrame:
    return _fetch_axs_rows(
        engine,
        start_date,
        "load_start_date",
        "load_country = :load_country "
        "AND voyage_type = :voyage_type "
        "AND COMMODITY LIKE :commodity",
        {
            "load_country": "Australia",
            "voyage_type": "laden",
            "commodity": "%COAL%",
        },
    )


def fetch_indonesia_shipments(
    engine: Engine, start_date: date | datetime | str
) -> pd.DataFrame:
    return _fetch_axs_rows(
        engine,
        start_date,
        "load_start_date",
        "COMMODITY LIKE :commodity AND load_country = :load_country",
        {"commodity": "%COAL%", "load_country": "Indonesia"},
    )


def fetch_china_arrivals(engine: Engine, start_date: date | datetime | str) -> pd.DataFrame:
    return _fetch_axs_rows(
        engine,
        start_date,
        "discharge_start_date",
        "COMMODITY LIKE :commodity "
        "AND discharge_country = :discharge_country "
        "AND load_country <> :excluded_load_country",
        {
            "commodity": "%COAL%",
            "discharge_country": "China",
            "excluded_load_country": "China",
        },
    )


def fetch_analysis_date_bounds(
    axs_engine: Engine,
    baltic_engine: Engine,
    baltic_schema: str = "market_data",
) -> tuple[date, date]:
    """Return the shared history available across all four dashboard series."""
    australia_bounds = _fetch_axs_date_bounds(
        axs_engine,
        "load_start_date",
        "load_country = :load_country "
        "AND voyage_type = :voyage_type "
        "AND COMMODITY LIKE :commodity",
        {
            "load_country": "Australia",
            "voyage_type": "laden",
            "commodity": "%COAL%",
        },
    )
    indonesia_bounds = _fetch_axs_date_bounds(
        axs_engine,
        "load_start_date",
        "COMMODITY LIKE :commodity AND load_country = :load_country",
        {"commodity": "%COAL%", "load_country": "Indonesia"},
    )
    china_bounds = _fetch_axs_date_bounds(
        axs_engine,
        "discharge_start_date",
        "COMMODITY LIKE :commodity "
        "AND discharge_country = :discharge_country "
        "AND load_country <> :excluded_load_country",
        {
            "commodity": "%COAL%",
            "discharge_country": "China",
            "excluded_load_country": "China",
        },
    )

    table = "baltic_indices"
    columns = _table_columns(baltic_engine, baltic_schema, table)
    date_col = _first_existing_column(columns, ("Date", "date"))
    name_col = _first_existing_column(columns, ("Name", "name"))
    if not date_col or not name_col:
        raise DataSourceError(
            f"Could not identify Date and Name columns in {baltic_schema}.{table}. "
            f"Available columns: {columns}"
        )

    baltic_query = text(
        f"""
        SELECT
          MIN({_quote_identifier(date_col)}) AS earliest_date,
          MAX({_quote_identifier(date_col)}) AS latest_date
        FROM {_quote_identifier(baltic_schema)}.{_quote_identifier(table)}
        WHERE {_quote_identifier(name_col)} = :name
        """
    )
    baltic_bounds = _read_date_bounds(
        pd.read_sql(baltic_query, baltic_engine, params={"name": "P3A_82"}),
        "Baltic P3A_82",
    )
    return _common_date_bounds(
        [australia_bounds, indonesia_bounds, china_bounds, baltic_bounds]
    )


def _find_baltic_value_candidates(engine: Engine, schema: str) -> pd.DataFrame:
    query = text(
        """
        SELECT TABLE_NAME, COLUMN_NAME
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = :schema
          AND (
            UPPER(COLUMN_NAME) LIKE '%P3A_82%'
            OR UPPER(COLUMN_NAME) LIKE '%P3A82%'
            OR UPPER(TABLE_NAME) LIKE '%P3A_82%'
            OR UPPER(TABLE_NAME) LIKE '%P3A82%'
          )
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
    )
    return pd.read_sql(query, engine, params={"schema": schema})


def discover_baltic_p3a82_source(engine: Engine, schema: str = "market_data") -> BalticSource:
    candidates = _find_baltic_value_candidates(engine, schema)
    if candidates.empty:
        raise DataSourceError(
            "Could not find a Baltic P3A_82 source in information_schema."
        )

    for record in candidates.to_dict("records"):
        table = str(record["TABLE_NAME"])
        column = str(record["COLUMN_NAME"])
        table_columns = _table_columns(engine, schema, table)
        date_col = _first_existing_column(table_columns, BALTIC_DATE_CANDIDATES)
        if not date_col:
            continue

        normalized_column = _normalize(column)
        if "p3a82" in normalized_column:
            value_col = column
        else:
            value_col = _first_existing_column(table_columns, BALTIC_VALUE_CANDIDATES)
        if value_col:
            return BalticSource(schema, table, date_col, value_col)

    raise DataSourceError(
        "Found possible P3A_82 tables or columns, but could not identify both "
        "a date column and a value column."
    )


def fetch_baltic_p3a82(
    engine: Engine, start_date: date | datetime | str, schema: str = "market_data"
) -> pd.DataFrame:
    table = "baltic_indices"

    columns = _table_columns(engine, schema, table)

    date_col = _first_existing_column(columns, ("Date", "date"))
    name_col = _first_existing_column(columns, ("Name", "name"))
    value_col = _first_existing_column(
        columns,
        (
            "Value",
            "value",
            "Price",
            "price",
            "Index",
            "index",
            "Close",
            "close",
            "Settle",
            "settle",
            "p3a_82",
        ),
    )

    if not date_col or not name_col or not value_col:
        raise DataSourceError(
            f"Could not identify Date, Name, and Value columns in {schema}.{table}. "
            f"Available columns: {columns}"
        )

    query = text(
        f"""
        SELECT
          {_quote_identifier(date_col)} AS date,
          {_quote_identifier(value_col)} AS value
        FROM {_quote_identifier(schema)}.{_quote_identifier(table)}
        WHERE {_quote_identifier(name_col)} = :name
          AND {_quote_identifier(date_col)} >= :start_date
        ORDER BY {_quote_identifier(date_col)}
        """
    )

    return pd.read_sql(
        query,
        engine,
        params={
            "name": "P3A_82",
            "start_date": _start_date_value(start_date),
        },
    )
