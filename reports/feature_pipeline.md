# FD001 Feature-Generation Pipeline

Owner: Sreetama Santra · Code: `src/features/build_features.py` · Tests: `tests/test_build_features.py`

## Purpose
Turn the raw NASA C-MAPSS FD001 training file into one reproducible, leakage-safe feature matrix
(`data/processed/features_fd001.csv`) plus a feature manifest (`reports/feature_manifest.csv`).
The pipeline reuses the team's existing modules instead of re-implementing them.

## Run
From the repository root, with `train_FD001.txt` in `data/raw/CMAPSSData/`:

    python -m src.features.build_features
    python -m pytest tests/test_build_features.py -v

The processed CSV is generated, not committed (`data/processed/` is git-ignored).
Reproducibility check: the command prints a SHA-256 of the output; two runs on the same machine must print the same value.

## Steps
| Step | Module | Notes |
|---|---|---|
| Load + validate raw file | `src.data.load_data` | 26 columns, numeric, no header |
| RUL and failure label | `src.data.target_labels` | `rul = final_cycle - cycle`; `target = 1` when `rul <= 24` |
| Engine-disjoint split | `src.data.temporal_split` | 70 / 15 / 15 engines; stored in the `split` column |
| Drop constant channels | `build_features.find_constant_columns` | Decided on **train engines only** |
| Lag features | `src.data.lag_features` | Lags 1, 5, 10 per kept sensor |
| Rolling mean/std/min/max | `src.features.rolling_features` | Windows 5, 10, 20 per kept sensor |

## Output layout
`unit_number, time_cycles, <model features...>, rul, target, split`

Model features = every column except `unit_number`, `time_cycles`, `rul`, `target`, `split`
(use `build_features.feature_columns(matrix)`). With the default settings: 242 features
(2 operating settings, 15 sensors, 45 lag, 45 each of rolling mean/std/min/max).

## Leakage controls
- Lag and rolling features at cycle *t* use cycles before *t* of the same engine only (the current cycle is excluded from rolling windows).
- Constant channels are chosen from training engines only.
- `rul`, `target`, `split` and ids are never model inputs.
- Tests check: shifted values equal earlier rows, changing future rows does not change earlier features, labels follow the rule, every engine sits in one split.

## Missing values
Warm-up rows are left as NaN, not filled: lag-*k* features are NaN for the first *k* cycles of each engine;
rolling features are NaN on cycle 1 and rolling std also on cycle 2. Imputation/scaling must be fitted on the training split only
(the logistic-regression baseline already does this inside its pipeline; XGBoost/LightGBM accept NaN directly).

## Assumptions and limitations
- Horizon of 24 is in operational cycles, not hours, and is pending team approval.
- FD001 has no raw vibration waveform, so no FFT features are produced.
- The split is engine-disjoint, not globally chronological (FD001 has no shared calendar time).
- Only the training file is processed. Test-file features (truncated histories) are not generated here.
- Lag and rolling features overlap heavily; see `notebooks/02_feature_quality.py` and `reports/selected_features_proposal.csv`
  for a redundancy-filtered proposal. It is a proposal to be confirmed on validation results.
