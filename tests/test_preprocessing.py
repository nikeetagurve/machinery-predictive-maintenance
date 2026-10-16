
import pandas as pd
import pytest

from src.data.preprocessing import (
    validate_time_series,
    fill_sensor_missing_values,
)


def test_validate_time_series_sorts_by_machine_and_time():
    df = pd.DataFrame({
        "machine_id": ["B", "A", "A"],
        "timestamp": [
            "2026-10-02",
            "2026-10-02",
            "2026-10-01",
        ],
        "temperature": [30, 20, 10],
    })

    result = validate_time_series(
        df, "machine_id", "timestamp"
    )

    assert result["machine_id"].tolist() == ["A", "A", "B"]
    assert result["timestamp"].is_monotonic_increasing is False


def test_fill_sensor_missing_values_uses_previous_value_per_machine():
    df = pd.DataFrame({
        "machine_id": ["A", "A", "B", "B"],
        "temperature": [10.0, None, 50.0, None],
    })

    result = fill_sensor_missing_values(
        df, "machine_id", ["temperature"]
    )

    assert result["temperature"].tolist() == [
        10.0, 10.0, 50.0, 50.0
    ]


def test_missing_required_column_raises_error():
    df = pd.DataFrame({
        "timestamp": ["2026-10-01"],
    })

    with pytest.raises(ValueError):
        validate_time_series(
            df, "machine_id", "timestamp"
        )
