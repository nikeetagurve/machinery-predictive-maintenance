
"""Leakage-safe rolling features for NASA C-MAPSS sensor data."""

import pandas as pd

DEFAULT_WINDOWS = (5, 10, 20)

ID_COLUMNS = ["unit_number", "time_cycles"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def add_rolling_features(
    df: pd.DataFrame,
    windows=DEFAULT_WINDOWS,
    sensor_columns=SENSOR_COLUMNS,
) -> pd.DataFrame:
    """Add rolling statistics using only previous cycles for each engine."""
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

    # Keep the original row order and index for the final output.
    result = df.copy()
    sorted_df = df.sort_values(
        ["unit_number", "time_cycles"], kind="stable"
    )

    # Store new features separately to avoid repeated DataFrame insertion.
    feature_data = {}

    for sensor in sensor_columns:
        grouped = sorted_df.groupby("unit_number", sort=False)[sensor]
        previous_values = grouped.shift(1)

        previous_groups = previous_values.groupby(
            sorted_df["unit_number"], sort=False
        )

        for window in windows:
            rolling = previous_groups.rolling(
                window=window, min_periods=1
            )

            stats = {
                "mean": rolling.mean(),
                "std": rolling.std(ddof=1),
                "min": rolling.min(),
                "max": rolling.max(),
            }

            for stat_name, values in stats.items():
                column_name = f"{sensor}_rolling_{stat_name}_{window}"

                # Remove the engine level from the rolling result index.
                values.index = values.index.droplevel(0)

                # Align to sorted_df's index; final output restores input order.
                feature_data[column_name] = values.reindex(sorted_df.index)

    features = pd.DataFrame(feature_data, index=sorted_df.index)
    sorted_result = pd.concat([sorted_df, features], axis=1)

    # Restore the original row order and index.
    result = sorted_result.loc[df.index]
    return result
