
# FD001 Target Labeling — October 20, 2026

## Objective
Create binary labels indicating whether an engine will fail within the next 24 operational cycles.

## Dataset
NASA C-MAPSS FD001 training dataset.

## Labeling method
For each engine:

Remaining Useful Life (RUL) = Final cycle - Current cycle

- Target = 1 when RUL <= 24
- Target = 0 when RUL > 24

## Assumptions
1. Each engine in the FD001 training dataset runs until failure.
2. The final observed training cycle represents the failure endpoint.
3. The prediction window is 24 operational cycles, not 24 hours.
4. Each engine is labeled independently.
5. The FD001 test dataset contains truncated engine histories, so its last observed cycle must not be treated as failure.
6. RUL is used to generate training labels and must not be included as a model input feature.

## Validation
Six target-labeling tests passed.

The full project test suite also passed: 32 tests.

## Limitations
The current function is intended for complete run-to-failure training histories. Additional handling is required for truncated test histories and live prediction data.
