
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

    return result.sort_values(
        [machine_col, timestamp_col]
    ).reset_index(drop=True)


def fill_sensor_missing_values(
    df: pd.DataFrame,
    machine_col: str,
    sensor_cols: list[str],
) -> pd.DataFrame:
    """Fill sensor gaps using past observations only."""
    if machine_col not in df.columns:
        raise ValueError(f"Missing machine column: {machine_col}")

    result = df.copy()

    for col in sensor_cols:
        if col not in result.columns:
            raise ValueError(f"Missing sensor column: {col}")

        result[col] = result.groupby(machine_col)[col].transform(
            lambda series: series.ffill()
        )

    return result


def validate_cycle_sequences(
    df: pd.DataFrame,
    machine_col: str,
    cycle_col: str,
) -> pd.DataFrame:
    """Validate unique, consecutive operational cycles per machine."""
    required = [machine_col, cycle_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    result = df.copy()
    if result[required].isna().any().any():
        raise ValueError("Machine IDs and cycles cannot be missing.")

    result[cycle_col] = pd.to_numeric(result[cycle_col], errors="raise")

    if (result[cycle_col] < 1).any():
        raise ValueError("Cycles must be positive.")

    if result.duplicated(required).any():
        raise ValueError("Duplicate machine-cycle pairs found.")

    result = result.sort_values(required).reset_index(drop=True)

    for machine, group in result.groupby(machine_col, sort=False):
        cycles = group[cycle_col].to_numpy()
        if len(cycles) > 1 and (cycles[1:] - cycles[:-1] != 1).any():
            raise ValueError(
                f"Non-consecutive cycles found for machine {machine}."
            )

    return result
