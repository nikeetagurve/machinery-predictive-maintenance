# FD001 Data Cleaning Decisions

## Dataset Summary
- Dataset: NASA C-MAPSS FD001 training data
- Input rows: 20,631
- Columns: 26
- Unique engines: 100
- Duplicate engine-cycle pairs: 0
- Missing cells: 0
- Output rows: 20,631

## Cleaning Decisions

1. Column names were assigned to the five engine/cycle/operating
   setting fields and the 21 sensor fields.
2. Data was validated for the expected 26 columns.
3. Missing values were checked. No missing cells were found, so
   imputation was unnecessary.
4. Duplicate engine-cycle pairs were checked. None were found.
5. Engine cycle sequences were validated to detect gaps or duplicates.
6. Rows were sorted by engine ID and cycle number.
7. No rows were deleted and no observations were invented.
8. Cycle number was preserved as the time index. No arbitrary
   calendar-time resampling was performed.
9. Sensor scaling and feature engineering are deferred to later
   modeling steps to avoid mixing preprocessing stages.
10. Raw and generated processed datasets should not be committed
    to Git; the cleaning script and documentation should be tracked.

## Reproducibility

Run:

    python3 -m src.data.clean_fd001

Validate the preprocessing tests:

    python3 -m pytest tests/test_preprocessing.py -v
