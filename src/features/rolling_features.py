
"""Leakage-safe rolling features for NASA C-MAPSS sensor data."""

import pandas as pd

DEFAULT_WINDOWS = (5, 10, 20)

# Engine identifiers, cycle number, operating settings aur sensors.
ID_COLUMNS = ["unit_number", "time_cycles"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def add_rolling_features(
    df: pd.DataFrame,
    windows=DEFAULT_WINDOWS,
    sensor_columns=SENSOR_COLUMNS,
) -> pd.DataFrame:
    """Add lagged rolling statistics without using current/future sensor readings.

    Each engine is processed independently. Statistics for cycle t use only
    observations from cycles strictly before t, preventing target leakage.
    The input DataFrame is not modified.
    """
    required = set(ID_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not df["unit_number"].notna().all():
        raise ValueError("unit_number cannot contain missing values.")

    if not df["time_cycles"].notna().all():
        raise ValueError("time_cycles cannot contain missing values.")

    if df.duplicated(["unit_number", "time_cycles"]).any():
        raise ValueError("Duplicate unit_number/time_cycles pairs found.")

    windows = tuple(windows)
    if not windows or any(
        not isinstance(w, int) or isinstance(w, bool) or w <= 0
        for w in windows
    ):
        raise ValueError("windows must contain positive integers.")

    if len(set(windows)) != len(windows):
        raise ValueError("windows must not contain duplicates.")

    sensor_columns = list(sensor_columns)
    missing_sensors = set(sensor_columns) - set(df.columns)
    if missing_sensors:
        raise ValueError(
            f"Sensor columns not found: {sorted(missing_sensors)}"
        )

    result = df.copy()
    original_index = result.index
    result = result.sort_values(
        ["unit_number", "time_cycles"], kind="stable"
    )

    for sensor in sensor_columns:
        grouped = result.groupby("unit_number", sort=False)[sensor]

        # shift(1) ensures that the current cycle is excluded.
        previous_values = grouped.shift(1)
        previous_groups = previous_values.groupby(
            result["unit_number"], sort=False
        )

        for window in windows:
            rolling = previous_groups.rolling(
                window=window, min_periods=1
            )

            features = {
                "mean": rolling.mean(),
                "std": rolling.std(ddof=1),
                "min": rolling.min(),
                "max": rolling.max(),
            }

            for stat_name, values in features.items():
                column_name = f"{sensor}_rolling_{stat_name}_{window}"
                # Rolling groupby results have a MultiIndex; remove engine level.
                values.index = values.index.droplevel(0)
                result[column_name] = values.reindex(result.index)

    # Restore original row order and index.
    result = result.loc[original_index]
    return result
