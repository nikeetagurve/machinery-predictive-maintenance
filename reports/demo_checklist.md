# Project Demo Checklist — Machinery Predictive Maintenance

## 1. Before the Demo
- [ ] Confirm the correct branch: feat/nikeeta-team-lead.
- [ ] Confirm Python dependencies are installed.
- [ ] Confirm NASA FD001 files are present in data/raw/CMAPSSData/.
- [ ] Confirm GRU artifacts are present in artifacts/.
- [ ] Run the automated test suite.
- [ ] Check that the final report and model comparison reports are available.

## 2. Demonstration Sequence

### Step 1 — Explain the Problem
Explain how predictive maintenance uses sensor readings and operating cycles to identify possible engine failure before it occurs.

### Step 2 — Introduce the Dataset
Show the NASA C-MAPSS FD001 dataset:
- 20,631 training rows.
- 13,096 test rows.
- 100 test engines.
- Three operating settings and 21 sensors.

### Step 3 — Explain the Pipeline
Present the workflow:
1. Data loading and validation.
2. Preprocessing.
3. Temporal feature engineering.
4. Model training and comparison.
5. GRU sequence prediction.
6. Alert generation.

### Step 4 — Present Model Results
Show:
- Engineered Logistic Regression F1-score: 0.8916.
- GRU validation F1-score: 0.8889.
- Precision, recall and confusion matrix.
- Explain why accuracy alone is insufficient for imbalanced data.

### Step 5 — Explain the Alert Threshold
Explain the provisional probability threshold of 0.5 for two consecutive cycles.
Clarify that the 24-cycle prediction horizon and maintenance cost assumptions require stakeholder approval.

### Step 6 — Run the Integrated Prediction
Run:

`python3 -m src.modeling.predict`

Explain the generated probability, NORMAL/ALERT status and prediction CSV.

Previously verified demonstration output:
- 10,196 predictions.
- 100 engines.
- 9,949 NORMAL predictions.
- 247 ALERT predictions.
- Zero missing values in the output CSV.

These are inference outputs, not confirmed real-world failures.

### Step 7 — Show Testing
Run:

`python3 -m pytest -q`

Previously verified result: 93 tests passed.

## 3. Key Files to Show
- `README.md`
- `reports/final_week4_report.md`
- `reports/model_comparison.md`
- `reports/gru_metrics.md`
- `reports/threshold_analysis.md`
- `reports/streaming_predictions.csv` (local generated output)
- `artifacts/gru_model.pth`
- `artifacts/gru_preprocessing.joblib`

## 4. Important Limitations to Mention
- The dataset is simulated turbofan engine data.
- Prediction horizon is measured in cycles, not hours.
- The alert threshold is provisional.
- The reserved test partition has not been used for final performance reporting.
- Actual MLflow run logs were not verified.
- Clean virtual-environment verification remains outstanding.
- The prototype is not yet production-ready.

## 5. Before Submission
- [ ] Confirm final report accuracy.
- [ ] Confirm README instructions and report links.
- [ ] Confirm requirements files.
- [ ] Confirm model artifacts and loading.
- [ ] Confirm all tests pass.
- [ ] Check Git diff and working tree.
- [ ] Confirm the Axlero review date and time.
- [ ] Commit only to feat/nikeeta-team-lead.
- [ ] Raise or update the pull request as instructed by the team.
