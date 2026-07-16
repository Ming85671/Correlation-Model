from datetime import date

from app import complete_history_years, max_selectable_months
from src.data_access import DataSourceError, _common_date_bounds


def test_common_date_bounds_uses_the_shared_overlap():
    result = _common_date_bounds(
        [
            (date(2010, 1, 1), date(2026, 6, 30)),
            (date(2012, 6, 1), date(2026, 7, 15)),
            (date(2011, 3, 1), date(2025, 12, 31)),
        ]
    )

    assert result == (date(2012, 6, 1), date(2025, 12, 31))


def test_common_date_bounds_rejects_non_overlapping_series():
    try:
        _common_date_bounds(
            [
                (date(2010, 1, 1), date(2011, 1, 1)),
                (date(2012, 1, 1), date(2026, 1, 1)),
            ]
        )
    except DataSourceError as exc:
        assert "overlapping" in str(exc)
    else:
        raise AssertionError("Expected non-overlapping ranges to raise DataSourceError")


def test_complete_history_years_counts_only_full_years():
    assert complete_history_years(date(2012, 6, 1), date(2025, 5, 31)) == 12
    assert complete_history_years(date(2012, 6, 1), date(2025, 6, 1)) == 13


def test_max_selectable_months_preserves_the_existing_full_year_limit():
    assert max_selectable_months(date(2012, 6, 1), date(2025, 5, 31)) == 144
