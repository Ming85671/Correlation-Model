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
    "quantity",
    "cargo_quantity",
    "cargo_qty",
    "cargo_volume",
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


def _fetch_axs_rows(
    engine: Engine,
    start_date: date | datetime | str,
    date_col: str,
    filters: str,
    params: dict[str, object] | None = None,
) -> pd.DataFrame:
    columns = _table_columns(engine, AXS_SCHEMA, AXS_TABLE)
    volume_col = _first_existing_column(columns, AXS_VOLUME_CANDIDATES)
    selected = [f"{_quote_identifier(date_col)} AS date"]
    if volume_col:
        selected.append(f"{_quote_identifier(volume_col)} AS volume")

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
    return pd.read_sql(query, engine, params=query_params)


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
