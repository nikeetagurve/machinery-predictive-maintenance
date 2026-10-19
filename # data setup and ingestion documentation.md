# data setup and ingestion documentation

## dataset overview
- source: nasa prognostics center of excellence turbofan engine degradation simulation
- target subsets: FD001, FD002, FD003, FD004
- raw format: whitespace-delimited space-separated text files without headers

## schema definition
- identifier: unit_nr
- cycle counter: time_cycles
- operational settings: setting_1, setting_2, setting_3
- sensor channels: s1 through s21
- total features per row: 26

## immutability policy
1. raw files placed in data/raw/ must remain read-only.
2. no transformations, updates, or formatting changes are written back to data/raw/.
3. consumption in modules must invoke src/data_loader.py to receive deep copies.

## validation checklist
every file loaded is verified against:
- column dimension parity strictly equal to 26
- complete numerical data types across all sensor and setting columns
- complete absence of null or na values
- unit and cycle values greater than or equal to 1

## usage instructions
place raw dataset files in:
data/raw/train_FD001.txt
data/raw/test_FD001.txt
data/raw/RUL_FD001.txt

run verification script:
python src/data_loader.py

run test suite:
pytest tests/test_data_loader.py