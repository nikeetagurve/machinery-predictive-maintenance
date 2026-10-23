
# FD001 Data Splitting Strategy

## Objective
Create reproducible training, validation, and testing
datasets while preventing data leakage.

## Dataset
NASA C-MAPSS FD001 simulated turbofan engine dataset.

Each engine has its own operating-cycle sequence.
Cycle numbers are not shared calendar timestamps.

## Splitting Strategy
The function divides complete engine histories into:

- Training: approximately 70% of engines
- Validation: approximately 15% of engines
- Testing: remaining engines

Engines are assigned deterministically using sorted IDs.

No engine appears in more than one split.

## Chronological Safety
Within each engine, observations are sorted by
operating cycle.

The existing validate_cycle_sequences() function
checks for missing, duplicate, and non-consecutive cycles.

The split prevents engine-history overlap between
training, validation, and testing datasets.

## Important Limitation
This is an engine-disjoint split, not a globally
chronological split.

FD001 does not contain a common calendar timestamp
that establishes chronological ordering across engines.

Sorting by engine ID does not establish real-world time order.

## Leakage Prevention
- Complete engine histories remain in one split.
- Lag and rolling features must use past readings only.
- Validation and test data must not be used to fit
  scalers, imputers, or feature selectors.
- The RUL and target columns must not be used as
  model input features.
- Hyperparameters should be selected using validation
  data, leaving test data for final evaluation.

## Reproducibility
The split is deterministic for the same input data
and ratio settings.

## Compatibility
The utility supports configurable engine and cycle
column names:

- Shankar's loader: unit_number, time_cycles
- Nikeeta's cleaner: engine_id, cycle

## Testing
Eight automated tests cover engine isolation,
row preservation, cycle ordering, determinism,
column compatibility, invalid ratios, duplicate
cycles, and input immutability.
