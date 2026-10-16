from src.data.preprocessing import validate_cycle_sequences
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
    assert result.loc[0, "timestamp"] < result.loc[1, "timestamp"]


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

def test_validate_cycle_sequences_sorts_and_preserves_data():
    df = pd.DataFrame({
        "machine_id": ["A", "A", "B"],
        "cycle": [2, 1, 1],
        "sensor": [20, 10, 50],
    })

    result = validate_cycle_sequences(df, "machine_id", "cycle")

    assert result["cycle"].tolist() == [1, 2, 1]
    assert result["sensor"].tolist() == [10, 20, 50]


def test_validate_cycle_sequences_rejects_gaps():
    df = pd.DataFrame({
        "machine_id": ["A", "A"],
        "cycle": [1, 3],
    })

    import pytest
    with pytest.raises(ValueError, match="Non-consecutive"):
        validate_cycle_sequences(df, "machine_id", "cycle")


def test_validate_cycle_sequences_rejects_duplicates():
    df = pd.DataFrame({
        "machine_id": ["A", "A"],
        "cycle": [1, 1],
    })

    import pytest
    with pytest.raises(ValueError, match="Duplicate"):
        validate_cycle_sequences(df, "machine_id", "cycle")

