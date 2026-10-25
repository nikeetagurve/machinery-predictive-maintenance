# Technical Rationale: Evaluation and Omission of Vibration/FFT Features

## 1. Context and Objective
Under the Week 2 feature engineering roadmap, the objective was to evaluate whether frequency-domain transformations—specifically Fast Fourier Transform (FFT), spectral density estimation, and harmonic frequency extraction—should be incorporated into the feature matrix for machinery degradation tracking.

## 2. Dataset Characteristics (NASA C-MAPSS)
The selected dataset (C-MAPSS turbofan degradation simulation) provides 21 sensor variables alongside 3 operational settings. The observation structure is characterized by:
- **Sampling Frequency & Aggregation:** Readings are logged per operational flight cycle (`time_cycles`), representing aggregated or steady-state snapshot values rather than raw, continuous time-series waveforms.
- **Physical Sensor Channels:** The telemetry captures thermodynamic and aerodynamic properties:
  - Total temperatures (T2, T24, T30, T50)
  - Total pressures (P2, P15, P30, Pb42)
  - Spool rotational speeds (`s9`: physical core speed, `s10`: corrected core speed, `s4`: physical fan speed, `s5`: corrected fan speed)
  - Bypass ratios, fuel flow ratios, and static pressures
- **Absence of High-Frequency Accelerometer Telemetry:** There are no raw high-frequency vibration sensors (e.g., piezoelectric accelerometers sampled at kHz ranges).

## 3. Technical Justification for Omission

1. **Violation of the Nyquist-Shannon Sampling Theorem for Vibration:**
   Vibration fault diagnosis (bearing fault pass frequencies, gear mesh frequencies, blade-passing frequencies) requires raw sampling rates typically exceeding 10 kHz to capture mechanical impacts and high-frequency resonances. C-MAPSS logs at 1 sample per flight cycle. Attempting to interpret inter-cycle fluctuations as vibrational frequencies violates Nyquist limits and leads to aliasing.

2. **Non-Uniform Temporal Spacing Across Cycles:**
   Flight cycles represent discrete operational missions of variable duration and varying operational regimes (especially in sub-datasets FD002 and FD004), rather than a strictly uniform sampling interval ($\Delta t$). Standard Discrete Fourier Transform (DFT/FFT) assumes uniform temporal sampling $x[n] = x(n\Delta t)$.

3. **Domain Degradation Dynamics:**
   In C-MAPSS, degradation manifests as slow monotonic trends and thermal drift across cycles (e.g., rising exhaust gas temperature `T50`, dropping pressure ratio `P30/P2`), not high-frequency spectral redistribution. These behaviors are best captured via:
   - Rolling statistical aggregations (mean, standard deviation, min, max)
   - Time-lag differences ($\Delta s_i = s_i[t] - s_i[t - k]$)
   - Exponential moving averages and degradation slope metrics

## 4. Alternative Feature Representation Adopted
In place of FFT frequency-domain bins, rotational speed channels (`s4`, `s9`, `s10`) and core pressures are tracked using temporal trend filters:
- Multi-scale rolling windows (windows: 5, 10, 20 cycles)
- Rate-of-change and acceleration differentials between successive cycle steps