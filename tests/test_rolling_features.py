
import pandas as pd
import pytest

from src.features.rolling_features import add_rolling_features


def sample_data():
    """Synthetic data for two engines; no real NASA data is used."""
    return pd.DataFrame({
        "unit_number": [1, 1, 1, 1, 2, 2],
        "time_cycles": [1, 2, 3, 4, 1, 2],
        "sensor_1": [10.0, 20.0, 30.0, 40.0, 100.0, 200.0],
    })


def test_rolling_feature_columns_are_created():
    result = add_rolling_features(
        sample_data(), windows=(2,), sensor_columns=["sensor_1"]
    )

    for stat in ("mean", "std", "min", "max"):
        assert f"sensor_1_rolling_{stat}_2" in result.columns


def test_uses_only_previous_cycles():
    result = add_rolling_features(
        sample_data(), windows=(2,), sensor_columns=["sensor_1"]
    )
    engine_1 = result[result["unit_number"] == 1].sort_values("time_cycles")

    # Cycle 1 has no previous observations.
    assert pd.isna(engine_1.iloc[0]["sensor_1_rolling_mean_2"])

    # Cycle 2 uses only cycle 1.
    assert engine_1.iloc[1]["sensor_1_rolling_mean_2"] == 10.0

    # Cycle 3 uses cycles 1 and 2, not its own value (30).
    assert engine_1.iloc[2]["sensor_1_rolling_mean_2"] == 15.0

    # Cycle 4 uses cycles 2 and 3.
    assert engine_1.iloc[3]["sensor_1_rolling_mean_2"] == 25.0


def test_engines_are_processed_independently():
    result = add_rolling_features(
        sample_data(), windows=(2,), sensor_columns=["sensor_1"]
    )
    engine_2 = result[
        (result["unit_number"] == 2) & (result["time_cycles"] == 1)
    ].iloc[0]

    # Engine 2's first cycle must not use engine 1's readings.
    assert pd.isna(engine_2["sensor_1_rolling_mean_2"])


def test_input_is_not_modified_and_original_order_is_preserved():
    original = sample_data().iloc[[3, 0, 5, 1, 4, 2]].copy()
    before = original.copy(deep=True)

    result = add_rolling_features(
        original, windows=(2,), sensor_columns=["sensor_1"]
    )

    pd.testing.assert_frame_equal(original, before)
    assert result.index.tolist() == original.index.tolist()
    assert result[["unit_number", "time_cycles"]].equals(
        original[["unit_number", "time_cycles"]]
    )


def test_duplicate_engine_cycle_pairs_are_rejected():
    df = pd.concat([sample_data(), sample_data().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        add_rolling_features(df, windows=(2,), sensor_columns=["sensor_1"])


def test_invalid_window_is_rejected():
    with pytest.raises(ValueError, match="positive integers"):
        add_rolling_features(
            sample_data(), windows=(0,), sensor_columns=["sensor_1"]
        )
