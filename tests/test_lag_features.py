
import pandas as pd
import pytest

from src.data.lag_features import create_lag_features


def test_lag_values_are_correct():
    df = pd.DataFrame({
        "engine_id": [1, 1, 1, 1],
        "cycle": [1, 2, 3, 4],
        "sensor_2": [10, 20, 30, 40],
    })

    result = create_lag_features(
        df, "engine_id", "cycle", ["sensor_2"], [1, 2]
    )

    assert pd.isna(result.loc[0, "sensor_2_lag_1"])
    assert result["sensor_2_lag_1"].iloc[1:].tolist() == [
        10, 20, 30
    ]
    assert result["sensor_2_lag_2"].iloc[2:].tolist() == [
        10, 20
    ]


def test_engines_do_not_mix():
    df = pd.DataFrame({
        "engine_id": [1, 1, 2, 2],
        "cycle": [1, 2, 1, 2],
        "sensor_2": [10, 20, 100, 200],
    })

    result = create_lag_features(
        df, "engine_id", "cycle", ["sensor_2"], [1]
    )

    assert pd.isna(result.loc[2, "sensor_2_lag_1"])
    assert result.loc[3, "sensor_2_lag_1"] == 100


def test_unsorted_rows_are_sorted_correctly():
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "cycle": [3, 1, 2],
        "sensor_2": [30, 10, 20],
    })

    result = create_lag_features(
        df, "engine_id", "cycle", ["sensor_2"], [1]
    )

    assert result["cycle"].tolist() == [1, 2, 3]
    assert result["sensor_2_lag_1"].iloc[2] == 20


def test_future_readings_are_not_used():
    df = pd.DataFrame({
        "engine_id": [1, 1, 1],
        "cycle": [1, 2, 3],
        "sensor_2": [10, 20, 999],
    })

    result = create_lag_features(
        df, "engine_id", "cycle", ["sensor_2"], [1]
    )

    # Cycle 2 must use Cycle 1, not future Cycle 3.
    assert result.loc[1, "sensor_2_lag_1"] == 10


def test_invalid_lag_raises_error():
    df = pd.DataFrame({
        "engine_id": [1, 1],
        "cycle": [1, 2],
        "sensor_2": [10, 20],
    })

    with pytest.raises(ValueError):
        create_lag_features(
            df, "engine_id", "cycle", ["sensor_2"], [-1]
        )


def test_missing_cycle_raises_error():
    df = pd.DataFrame({
        "engine_id": [1, 1],
        "cycle": [1, 3],
        "sensor_2": [10, 30],
    })

    with pytest.raises(ValueError, match="Non-consecutive"):
        create_lag_features(
            df, "engine_id", "cycle", ["sensor_2"], [1]
        )


def test_original_dataframe_is_unchanged():
    df = pd.DataFrame({
        "engine_id": [1, 1],
        "cycle": [1, 2],
        "sensor_2": [10, 20],
    })
    original = df.copy(deep=True)

    create_lag_features(
        df, "engine_id", "cycle", ["sensor_2"], [1]
    )

    pd.testing.assert_frame_equal(df, original)
