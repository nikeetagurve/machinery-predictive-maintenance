# Dataset Selection and Target Definition

**Project:** Axlero Project 2 — Machinery Predictive Maintenance  
**Prepared by:** Veda M V (Member 2)  
**Date:** 11 October 2026  
**Decision status:** Approved by the team on 11 October 2026

## 1. Recommended dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset — FD001 subset**

- **Official source:** https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- **NASA Prognostics Data Repository:** https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **Dataset type:** Simulated, multivariate engine sensor time series, indexed by engine ID and operational cycle.
- **Why FD001:** A comparatively simple starting subset with one operating condition and one fault mode; it provides run-to-failure training trajectories suitable for temporal feature engineering and advance-failure labeling.

## 2. Initial inspection evidence

Inspection performed locally in Python/Pandas on `train_FD001.txt`:

| Property | Observed value |
|---|---:|
| Training rows | 20,631 |
| Columns | 26 |
| Unique training engines | 100 |
| Missing cells (after parsing) | 0 |

The 26 columns comprise engine ID, operational cycle, three operational settings, and 21 sensor measurements. Training engines have differing final cycles; for example, engine 1 ends at cycle 192 and engine 2 at cycle 287. These training trajectories are treated as run-to-failure histories. The test trajectories are truncated and accompanied by separate RUL labels; do not infer their failure cycles merely from the last test row.

## 3. Proposed prediction target

**Task type:** Binary classification of imminent failure.

**Initial prediction horizon:** 24 **operational cycles** (not 24 hours). This is a provisional project choice to discuss with the team, not a claim that a cycle is an hour or that 24 is an empirically optimal horizon.

For each engine in the run-to-failure training subset:

- `final_cycle = max(cycle)` for that engine.
- `RUL = final_cycle - cycle`.
- `failure_within_24_cycles = 1` when `RUL <= 24`; otherwise `0`.

Using this rule on `train_FD001.txt` yielded:

| Target | Rows | Share |
|---|---:|---:|
| 0 — more than 24 cycles remain | 18,131 | 87.9% |
| 1 — 24 or fewer cycles remain | 2,500 | 12.1% |

**Important leakage rule:** `final_cycle`, `RUL`, and derived labels are supervision/analysis fields, **not model input features**. Any features must be constructed only from information available up to the prediction cycle. Split strategy should avoid leakage across engines and future observations, with an explicit documented evaluation protocol.

**Evaluation note:** A positive label inside a 24-cycle window does not prove that the trained model gives 24 cycles of advance notice. Measure actual warning lead time, precision, recall, false alarms, and F1 on held-out engines/sequences.

## 4. Dataset source, license, and attribution

The dataset is publicly downloadable from NASA's repository. **An explicit license was not identified in the dataset listing reviewed for this proposal.** Do not assume that public download means CC0, MIT, or another specific license. Before redistribution or including raw files in the public team repository, confirm applicable NASA dataset usage terms and the team's intended use. Cite NASA and the C-MAPSS dataset in the README/report. Prefer documenting download instructions rather than committing raw data.

## 5. Fit and limitations

**Fit:** Engine-specific sequences support lag features, rolling-window statistics, time-aware evaluation, tree models, and simulated row-by-row prediction.

**Limitations:** Data is simulated aircraft-engine degradation, not actual factory sensor telemetry. The operational cycle is not a timestamp in hours. FD001 does not provide raw high-frequency vibration waveforms for a conventional FFT experiment; the team should document whether FFT is appropriate. The 24-cycle horizon and evaluation strategy require team approval.

## 6. Team decisions required

- [ ] Approve NASA C-MAPSS FD001 as the shared dataset.
- [ ] Approve or revise the 24-cycle warning horizon and document the rationale.
- [ ] Confirm license/usage conditions before redistributing any dataset files.
- [ ] Agree on the held-out engine/time evaluation protocol.
- [ ] Share the source and labeling convention with Members 3–6 before feature engineering and modeling.

**Prepared for the October 11 dataset decision checkpoint.**
