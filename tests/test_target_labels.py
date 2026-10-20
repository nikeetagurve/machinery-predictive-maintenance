
import pandas as pd
import pytest

from src.data.target_labels import create_failure_labels


def test_failure_labels():
    # Engine 1 fails at cycle 30
    df = pd.DataFrame({
        "unit_number": [1, 1, 1, 1],
        "time_cycles": [1, 5, 6, 30],
    })

    result = create_failure_labels(df, prediction_window=24)

    assert result["rul"].tolist() == [29, 25, 24, 0]
    assert result["target"].tolist() == [0, 0, 1, 1]


def test_multiple_engines():
    df = pd.DataFrame({
        "unit_number": [1, 1, 2, 2],
        "time_cycles": [1, 30, 1, 50],
    })

    result = create_failure_labels(df, prediction_window=24)

    assert result["rul"].tolist() == [29, 0, 49, 0]
    assert result["target"].tolist() == [0, 1, 0, 1]


def test_missing_column():
    df = pd.DataFrame({
        "unit_number": [1, 1],
    })

    with pytest.raises(ValueError):
        create_failure_labels(df)


def test_duplicate_cycles():
    df = pd.DataFrame({
        "unit_number": [1, 1],
        "time_cycles": [5, 5],
    })

    with pytest.raises(ValueError):
        create_failure_labels(df)


def test_original_data_unchanged():
    df = pd.DataFrame({
        "unit_number": [1, 1],
        "time_cycles": [1, 30],
    })

    original = df.copy(deep=True)

    create_failure_labels(df)

    pd.testing.assert_frame_equal(df, original)


def test_invalid_window():
    df = pd.DataFrame({
        "unit_number": [1, 1],
        "time_cycles": [1, 30],
    })

    with pytest.raises(ValueError):
        create_failure_labels(df, prediction_window=0)


@pytest.mark.parametrize("invalid_cycle", [-5, 0, 2.5])
def test_invalid_cycle_numbers(invalid_cycle):
    df = pd.DataFrame({
        "unit_number": [1, 1],
        "time_cycles": [1, invalid_cycle],
    })

    with pytest.raises(ValueError):
        create_failure_labels(df)


def test_non_numeric_cycles():
    df = pd.DataFrame({
        "unit_number": [1, 1],
        "time_cycles": [1, "abc"],
    })

    with pytest.raises(ValueError):
        create_failure_labels(df)
