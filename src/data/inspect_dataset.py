"""Small, read-only helper functions for inspecting the FD001 data.

Nothing here removes, fills or changes any data.
"""
import numpy as np
import pandas as pd

from src.data.load_data import COLUMN_NAMES

UNVERIFIED = "Not specified/verified"

# UNCONFIRMED candidate names, in the listed order of Table 2 in
# Saxena et al. (2008), "Damage Propagation Modeling for Aircraft Engine
# Run-to-Failure Simulation". The paper does NOT say which sensor_N column
# matches which row, so these are only shown in the 'notes' column.
SENSOR_CANDIDATES = [
    "T2 - Total temperature at fan inlet (deg R)",
    "T24 - Total temperature at LPC outlet (deg R)",
    "T30 - Total temperature at HPC outlet (deg R)",
    "T50 - Total temperature at LPT outlet (deg R)",
    "P2 - Pressure at fan inlet (psia)",
    "P15 - Total pressure in bypass-duct (psia)",
    "P30 - Total pressure at HPC outlet (psia)",
    "Nf - Physical fan speed (rpm)",
    "Nc - Physical core speed (rpm)",
    "epr - Engine pressure ratio (no unit given)",
    "Ps30 - Static pressure at HPC outlet (psia)",
    "phi - Ratio of fuel flow to Ps30 (pps/psi)",
    "NRf - Corrected fan speed (rpm)",
    "NRc - Corrected core speed (rpm)",
    "BPR - Bypass ratio (no unit given)",
    "farB - Burner fuel-air ratio (no unit given)",
    "htBleed - Bleed enthalpy (no unit given)",
    "Nf_dmd - Demanded fan speed (rpm)",
    "PCNfR_dmd - Demanded corrected fan speed (rpm)",
    "W31 - HPT coolant bleed (lbm/s)",
    "W32 - LPT coolant bleed (lbm/s)",
]


def missing_value_table(df):
    """Missing-value count and percentage for every column."""
    counts = df.isna().sum()
    return pd.DataFrame({"missing_count": counts, "missing_percentage": (100 * counts / len(df)).round(4)})


def count_duplicates(df):
    """Count repeated full rows and repeated (unit_number, time_cycles) pairs."""
    return {
        "full_row_duplicates": int(df.duplicated().sum()),
        "unit_time_duplicates": int(df.duplicated(subset=["unit_number", "time_cycles"]).sum()),
    }


def count_infinite(df):
    """Count +/- infinity values in the whole table."""
    return int(np.isinf(df.to_numpy(dtype=float)).sum())


def engine_cycle_summary(df):
    """One row per engine: first cycle, last cycle, number of rows, and whether cycles are consecutive."""
    summary = df.groupby("unit_number")["time_cycles"].agg(first_cycle="min", last_cycle="max", n_rows="count")
    summary["consecutive"] = (summary["last_cycle"] - summary["first_cycle"] + 1) == summary["n_rows"]
    return summary


def find_constant_columns(df):
    """Operational settings / sensors that have only one distinct value."""
    channels = [c for c in df.columns if c.startswith(("op_setting_", "sensor_"))]
    return [c for c in channels if df[c].nunique() <= 1]


def check_rul_count(test, rul):
    """The RUL file should have one value per test engine."""
    n_engines = int(test["unit_number"].nunique())
    return {"test_engines": n_engines, "rul_values": int(len(rul)), "match": n_engines == len(rul)}


def make_data_dictionary(train=None):
    """Build the 26-row data dictionary. Without data, observed fields are left as 'pending'/blank."""
    rows = []
    for name in COLUMN_NAMES:
        if name == "unit_number":
            category, desc, unit, notes = "identifier", "Engine (unit) identifier", "Not applicable (identifier)", "NASA: column 1 = unit number."
        elif name == "time_cycles":
            category, desc, unit, notes = "time index", "Time in operational cycles", "cycles", "NASA: column 2 = time, in cycles."
        elif name.startswith("op_setting_"):
            n = name.split("_")[-1]
            category, unit = "operational setting", UNVERIFIED
            desc = f"Operational setting {n}. Meaning: {UNVERIFIED} in NASA documentation."
            notes = "NASA: three operational settings; FD001 has one operating condition (sea level)."
        else:
            n = int(name.split("_")[-1])
            category, unit = "sensor measurement", UNVERIFIED
            desc = f"Sensor measurement {n}. Meaning: {UNVERIFIED} in NASA documentation."
            notes = "UNCONFIRMED candidate (Saxena et al. 2008, Table 2, listed order): " + SENSOR_CANDIDATES[n - 1]

        if train is None:
            dtype, missing, pct, unique = "pending", "", "", ""
            notes = "[TEMPLATE - run the notebook] " + notes
        else:
            missing = int(train[name].isna().sum())
            dtype, pct, unique = str(train[name].dtype), round(100 * missing / len(train), 4), int(train[name].nunique())
        rows.append([name, category, desc, dtype, unit, missing, pct, unique, notes])

    columns = ["column_name", "category", "description", "observed_dtype", "unit",
               "missing_count", "missing_percentage", "unique_count", "notes"]
    return pd.DataFrame(rows, columns=columns)


def dataframe_to_markdown(df, index=True):
    """Turn a small DataFrame into a Markdown table (no extra packages needed)."""
    if index:
        df = df.reset_index()
    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |", "| " + " | ".join("---" for _ in df.columns) + " |"]
    for _, row in df.iterrows():
        cells = [f"{v:.5g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
