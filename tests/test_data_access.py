import pandas as pd

from src.data_access import (
    AXS_VOLUME_CANDIDATES,
    _axs_volume_columns,
    _best_volume_column,
    _find_axs_volume_column,
    _first_existing_column,
    _volume_candidate_stats,
)


def test_volume_column_detection_supports_cargo_tonnage_names():
    assert (
        _first_existing_column(["date", "cargo_tonnage"], AXS_VOLUME_CANDIDATES)
        == "cargo_tonnage"
    )


def test_volume_column_detection_supports_source_specific_unit_suffixes():
    assert _find_axs_volume_column(["date", "cargo_volume_mt"]) == "cargo_volume_mt"


def test_volume_column_detection_considers_all_matching_columns():
    assert _axs_volume_columns(["quantity", "cargo_volume_mt"]) == [
        "quantity",
        "cargo_volume_mt",
    ]


def test_best_volume_column_ignores_empty_or_constant_candidates():
    rows = pd.DataFrame(
        {
            "volume_0": [None, None, None],
            "volume_1": [0, 0, 0],
            "volume_2": ["10,000 MT", "15,000 MT", "20,000 MT"],
        }
    )

    assert _best_volume_column(rows, ["volume_0", "volume_1", "volume_2"]) == "volume_2"


def test_volume_candidate_stats_reports_numeric_and_distinct_counts():
    rows = pd.DataFrame({"volume_0": ["10,000 MT", None, "10,000 MT"]})

    assert _volume_candidate_stats(rows, ["volume_0"]) == [
        {"column": "volume_0", "numeric_observations": 2, "distinct_values": 1}
    ]
