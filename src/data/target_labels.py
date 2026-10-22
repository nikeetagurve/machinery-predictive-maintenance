
import pandas as pd

def create_failure_labels(
    df,
    machine_col="unit_number",
    cycle_col="time_cycles",
    prediction_window=24,
):
    """
    Label training rows based on whether an engine
    will fail within the next prediction_window cycles.

    0 = Failure is more than 24 cycles away
    1 = Failure is within 24 cycles
    """
    if prediction_window <= 0 or not isinstance(prediction_window, int):
        raise ValueError("Prediction window must be a positive integer.")

    required_cols = [machine_col, cycle_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if df[required_cols].isna().any().any():
        raise ValueError("Engine IDs and cycle numbers cannot be missing.")
    cycles = df[cycle_col]

    if not pd.api.types.is_numeric_dtype(cycles):
        raise ValueError("Cycle numbers must be numeric.")

    if not (cycles > 0).all():
        raise ValueError("Cycle numbers must be positive.")

    if not (cycles % 1 == 0).all():
        raise ValueError("Cycle numbers must be whole numbers.")

    if df.duplicated(subset=required_cols).any():
        raise ValueError("Duplicate engine-cycle combinations found.")

    result = df.copy()

    final_cycles = result.groupby(machine_col)[cycle_col].transform("max")

    result["rul"] = final_cycles - result[cycle_col]

    result["target"] = (
        result["rul"] <= prediction_window
    ).astype(int)

    return result