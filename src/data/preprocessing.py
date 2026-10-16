
import pandas as pd


def validate_time_series(
    df: pd.DataFrame,
    machine_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """Validate and chronologically sort machine sensor observations."""
    required = [machine_col, timestamp_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()
    result[timestamp_col] = pd.to_datetime(
        result[timestamp_col], errors="raise"
    )

    if result[timestamp_col].isna().any():
        raise ValueError("Timestamps contain missing values.")

    result = result.sort_values(
        [machine_col, timestamp_col]
    ).reset_index(drop=True)

    return result


def fill_sensor_missing_values(
    df: pd.DataFrame,
    machine_col: str,
    sensor_cols: list[str],
) -> pd.DataFrame:
    """Fill sensor gaps using past observations only."""
    result = df.copy()

    for col in sensor_cols:
        if col not in result.columns:
            raise ValueError(f"Missing sensor column: {col}")

        result[col] = result.groupby(machine_col)[col].transform(
            lambda series: series.ffill()
        )

    return result
v
