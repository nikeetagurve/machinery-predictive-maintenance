
import pandas as pd

from src.data.preprocessing import validate_cycle_sequences


def create_lag_features(
    df: pd.DataFrame,
    machine_col: str,
    cycle_col: str,
    sensor_cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Create past-cycle sensor features for each machine."""

    if not sensor_cols:
        raise ValueError("At least one sensor is required.")

    if not lags or any(
        type(lag) is not int or lag <= 0 for lag in lags
    ):
        raise ValueError("Lags must be positive integers.")

    if len(set(lags)) != len(lags):
        raise ValueError("Duplicate lag values are not allowed.")

    if len(set(sensor_cols)) != len(sensor_cols):
        raise ValueError("Duplicate sensor columns are not allowed.")

    missing = [col for col in sensor_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor columns: {missing}")

    # Validate and sort each engine's cycles.
    result = validate_cycle_sequences(
        df, machine_col, cycle_col
    )

    for sensor in sensor_cols:
        if not pd.api.types.is_numeric_dtype(result[sensor]):
            raise ValueError(f"Sensor must be numeric: {sensor}")

        for lag in lags:
            new_col = f"{sensor}_lag_{lag}"

            if new_col in result.columns:
                raise ValueError(f"Column already exists: {new_col}")

            # Get previous readings from the SAME engine.
            result[new_col] = (
                result.groupby(machine_col)[sensor].shift(lag)
            )

    return result
