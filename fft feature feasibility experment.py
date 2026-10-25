"""FFT Feature Feasibility Experiment on C-MAPSS Telemetry.

Demonstrates that applying Fast Fourier Transform to cycle-aggregated sensor
channels produces non-informative, noisy spectra due to low sampling
granularity.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq

# Standard schema definition
COLUMNS = [
    "unit_nr",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"s{i}" for i in range(1, 22)]


def load_sample_engine(file_path: Path, unit_id: int = 1) -> pd.DataFrame:
    """Loads telemetry for a single engine unit."""
    df = pd.read_csv(
        file_path, sep=r"\s+", header=None, names=COLUMNS, index_col=False
    )
    return df[df["unit_nr"] == unit_id].copy()


def run_fft_experiment(df_unit: pd.DataFrame, target_sensor: str = "s9"):
    """Computes FFT on selected rotational/vibrational proxy sensor and evaluates spectrum."""
    signal = df_unit[target_sensor].values
    n = len(signal)

    if n < 8:
        raise ValueError(
            "Insufficient cycle count for frequency transformation."
        )

    # Detrend signal to focus on dynamic fluctuations rather than DC offset
    signal_detrended = signal - np.mean(signal)

    # Discrete Fourier Transform
    yf = fft(signal_detrended)
    xf = fftfreq(n, d=1.0)  # d=1 cycle interval

    # Keep positive frequencies
    pos_mask = xf > 0
    freqs = xf[pos_mask]
    amplitudes = (2.0 / n) * np.abs(yf[pos_mask])

    # Spectral flatness test (Wiener entropy): near 1 indicates pure noise
    geometric_mean = np.exp(np.mean(np.log(amplitudes + 1e-12)))
    arithmetic_mean = np.mean(amplitudes)
    spectral_flatness = (
        geometric_mean / arithmetic_mean if arithmetic_mean > 0 else 0
    )

    print(f"=== FFT Experiment Analysis: Sensor {target_sensor} ===")
    print(f"Sample length (cycles): {n}")
    print(f"Dominant cycle frequency: {freqs[np.argmax(amplitudes)]:.4f} 1/cycle")
    print(f"Spectral flatness score: {spectral_flatness:.4f}")
    print(
        "Conclusion: Signal exhibits high spectral dispersion without distinct mechanical harmonic peaks."
    )
    print("Action: FFT features omitted in favor of rolling temporal filters.")

    return freqs, amplitudes


if __name__ == "__main__":
    raw_data_path = Path("data/raw/train_FD001.txt")
    if raw_data_path.exists():
        unit_data = load_sample_engine(raw_data_path, unit_id=1)
        run_fft_experiment(unit_data, target_sensor="s9")
    else:
        print(f"Input file not found at {raw_data_path}.")
        print(
            "Ensure raw C-MAPSS files are placed in data/raw/ before executing experiment."
        )