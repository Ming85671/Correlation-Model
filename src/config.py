from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class SecretsConfigError(RuntimeError):
    """Raised when required database settings are missing."""


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    database: str
    user: str
    password: str


def _get_section(secrets: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    try:
        value = secrets[section]
    except KeyError as exc:
        raise SecretsConfigError(
            f"Missing Streamlit secrets section [{section}]."
        ) from exc

    if not isinstance(value, Mapping):
        raise SecretsConfigError(f"Streamlit secrets section [{section}] is invalid.")
    return value


def get_database_settings(
    secrets: Mapping[str, Any], section: str
) -> DatabaseSettings:
    values = _get_section(secrets, section)
    missing = [
        key
        for key in ("host", "database", "user", "password")
        if not str(values.get(key, "")).strip()
    ]
    if missing:
        joined = ", ".join(missing)
        raise SecretsConfigError(
            f"Missing required Streamlit secrets in [{section}]: {joined}."
        )

    return DatabaseSettings(
        host=str(values["host"]).strip(),
        database=str(values["database"]).strip(),
        user=str(values["user"]).strip(),
        password=str(values["password"]),
    )
