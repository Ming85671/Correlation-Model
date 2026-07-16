from src.data_access import AXS_VOLUME_CANDIDATES, _find_axs_volume_column, _first_existing_column


def test_volume_column_detection_supports_cargo_tonnage_names():
    assert (
        _first_existing_column(["date", "cargo_tonnage"], AXS_VOLUME_CANDIDATES)
        == "cargo_tonnage"
    )


def test_volume_column_detection_supports_source_specific_unit_suffixes():
    assert _find_axs_volume_column(["date", "cargo_volume_mt"]) == "cargo_volume_mt"
