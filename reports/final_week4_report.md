# Week 4 Final Report — Machinery Predictive Maintenance

## 1. Executive Summary

This project develops a predictive maintenance pipeline using the NASA C-MAPSS FD001 simulated turbofan engine dataset. The objective is to identify engine conditions associated with an approaching failure and provide an early warning to support maintenance planning.

The integrated project includes data validation, preprocessing, temporal feature engineering, model comparison, GRU sequence modelling, provisional alert-threshold analysis, automated tests, and streaming inference.

Integration verification completed successfully in the development environment:
- All 93 automated tests passed.
- The NASA FD001 train, test, and RUL files loaded successfully.
- The saved GRU model and preprocessing artifact loaded successfully.
- Streaming inference generated 10,196 predictions across 100 test engines.
- The generated prediction CSV contained no missing values.

The alert threshold and prediction horizon remain provisional and require stakeholder approval before operational use.

## 2. Dataset and Objective

**Dataset:** NASA C-MAPSS FD001 simulated turbofan engine data.

| Dataset file | Shape |
|---|---:|
| Training data | 20,631 rows × 26 columns |
| Test data | 13,096 rows × 26 columns |
| Test RUL labels | 100 rows × 1 column |

The train and test files contain engine identifiers, operating cycles, three operating settings, and 21 sensor measurements.

**Prediction objective:** Classify whether an engine is approaching failure within the next 24 operational cycles, using the provisional target definition RUL <= 24.

The 24-cycle prediction horizon is an assumption for this project, not an approved maintenance policy. A cycle is not equivalent to an hour or a fixed amount of calendar time.

## 3. Data Validation and Leakage Prevention

The data loader checks that required files exist, are non-empty, contain numeric values, and have the expected number of columns.

The project uses an engine-disjoint 70/15/15 train, validation, and test split. Engines are kept separate across the partitions to reduce leakage from observations belonging to the same engine.

Additional safeguards include:
- Preprocessing is fitted on training data only.
- Temporal lag and rolling features use past readings.
- The test partition is reserved for final evaluation and is not used for model selection or threshold tuning.

The FD001 test data contains 100 engines. The final test partition contains 15 engines under the documented split.

## 4. Feature Engineering

The feature pipeline creates temporal information from sensor readings to capture changes in engine behaviour.

The documented feature workflow includes:
- Lag features at 1, 5, and 10 cycles.
- Rolling mean, standard deviation, minimum, and maximum over windows of 5, 10, and 20 cycles.
- Operating settings and sensor measurements.
- A total of 242 engineered features, with a filtered candidate set of 166 features.

FFT-based features were not used because this dataset does not provide raw vibration waveforms.

## 5. Model Comparison

The main model comparison uses 2,775 eligible validation rows from 15 engines: 375 positive and 2,400 negative rows. The reported comparison uses a probability threshold of 0.5.

| Model | Precision | Recall | F1-score | Accuracy | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression — engineered | 0.8284 | 0.9653 | **0.8916** | 0.9683 | 0.9760 |
| Gradient Boosting — filtered engineered | 0.8241 | 0.9493 | 0.8823 | 0.9658 | 0.9739 |
| Gradient Boosting — engineered | 0.8329 | 0.9307 | 0.8791 | 0.9654 | 0.9742 |
| Logistic Regression — raw | 0.6583 | 0.9760 | 0.7863 | 0.9283 | 0.9495 |

The always-negative baseline achieved 0.8649 accuracy but did not identify positive cases. This demonstrates why accuracy alone is not an adequate model-selection measure for this imbalanced dataset.

### Interpretation

Temporal feature engineering substantially improved the Logistic Regression result: F1 increased from 0.7863 with raw features to 0.8916 with engineered features. The raw model produced 190 false positives, compared with 75 for the engineered model.

The engineered Logistic Regression model achieved the highest F1-score in the main comparison. The interim Gradient Boosting model used HistGradientBoosting rather than a final XGBoost or LightGBM implementation.

## 6. GRU Model Results

The project also includes a one-layer GRU sequence model.

Documented configuration:
- Sequence length: 30 cycles.
- Inputs: three operating settings and 21 sensor features.
- Hidden size: 32.
- Batch size: 64.
- Training epochs: 20.
- Optimizer: Adam, learning rate 0.001.
- Loss: weighted binary cross-entropy with logits.
- Evaluation threshold: 0.5.

Validation results:

| Metric | GRU |
|---|---:|
| Precision | 0.8276 |
| Recall | 0.9600 |
| F1-score | 0.8889 |
| Accuracy | 0.9676 |

Confusion matrix:
- True negatives: 2,325
- False positives: 75
- False negatives: 15
- True positives: 360

In the separate comparison against raw-feature Logistic Regression, GRU improved F1, precision, and accuracy, while Logistic Regression had slightly higher recall.

The GRU result and the engineered Logistic Regression result come from separate comparison reports. Their small F1 difference should not be treated as proof of a definitive winner. The main model comparison also records that GRU was not rerun in that particular comparison.

## 7. Alert Threshold and Decision Analysis

The threshold analysis uses training-engine out-of-fold predictions for decision-making and a held-out validation confirmation.

The provisional recommendation is:
- Probability threshold: 0.5.
- Persistence: alert after the threshold is reached for two consecutive cycles.
- Prediction horizon: 24 cycles.
- Minimum notice assumption: 10 cycles.

The analysis considered the cost of missed failures relative to false alarms and tested robustness across five Logistic Regression regularization settings.

On the held-out validation engines, the documented two-consecutive-cycle rule produced:
- Zero missed engines.
- Zero too-late warnings.
- One far-premature warning.
- Minimum warning time of 19 cycles.
- Row-level precision of 0.868, recall of 0.947, and F1-score of 0.906.

These results are specific to the analysed model and assumptions. The threshold is **provisional, not an approved production setting**. The cost ratio, acceptable early-warning window, and 24-cycle horizon require confirmation with the project stakeholders. A threshold calibrated for Logistic Regression must not automatically be applied to the GRU model.

## 8. Integration and End-to-End Verification

The integrated project was checked in the development environment.

### Automated tests

Command:
`python3 -m pytest -q`

Result: **93 passed in 10.32 seconds.**

### Dataset loading

The loader successfully read:
- Training data: 20,631 × 26.
- Test data: 13,096 × 26.
- RUL labels: 100 × 1.
- Test engines: 100.

### GRU streaming inference

Command:
`python3 -m src.modeling.predict`

The saved model and preprocessing artifact loaded, and streaming inference completed.

| Streaming output | Result |
|---|---:|
| Predictions | 10,196 |
| Engines | 100 |
| NORMAL | 9,949 |
| ALERT | 247 |
| Missing values in CSV | 0 |

Output file: `reports/streaming_predictions.csv`

The CSV is generated locally and ignored by Git. These streaming alert counts are inference outputs, not an independent measure of model accuracy or confirmed real-world failures.

### Environment

PyTorch 2.15.0 was installed successfully. The Mac environment reported MPS availability, although the current prediction implementation selects CUDA when available and otherwise CPU.

The 93-test result and streaming run were verified in the existing Anaconda environment. A fresh virtual-environment installation and end-to-end run have not yet been verified.

## 9. Experiment Tracking and Reproducibility Status

The repository contains documentation describing an MLflow tracking setup. However, the verification performed for this report did not find MLflow run files under `reports`, and MLflow was not installed in the checked environment.

Therefore, actual MLflow experiment-run logging is **not verified**.

The GRU model and preprocessing artifacts are present:
- `artifacts/gru_model.pth`
- `artifacts/gru_preprocessing.joblib`

Their loading was verified through the streaming inference run.

A dedicated `model_io.py`, `reproduce_model.py`, manifest JSON, or MLflow `MLmodel` file was not found in the inspected project paths. These should be treated as outstanding reproducibility work rather than completed functionality.

## 10. Business Interpretation

A predictive maintenance system can help maintenance teams identify potentially deteriorating engines before failure, giving them an opportunity to inspect and schedule maintenance.

The decision involves two competing risks:
- A missed or late warning may lead to an unexpected failure and costly downtime.
- A false or premature warning may cause unnecessary inspection, maintenance, or operational disruption.

Precision, recall, PR-AUC, warning lead time, and engine-level outcomes are therefore important alongside accuracy. The provisional persistence rule is intended to avoid reacting to a single isolated high-probability reading.

The project demonstrates a predictive workflow on simulated engine data. It does not establish real-world savings, deployment readiness, or a validated maintenance policy.

## 11. Limitations and Future Work

1. C-MAPSS FD001 is simulated turbofan data and may not represent every real industrial operating environment.
2. The prediction horizon is expressed in cycles, not hours or calendar time.
3. The validation set contains only 15 engines, so engine-level conclusions have limited sample support.
4. The held-out test partition has been reserved; independent final test performance has not been reported here.
5. The threshold analysis depends on provisional cost and warning-time assumptions.
6. Thresholds are model-specific and require separate calibration for GRU deployment.
7. Actual MLflow experiment logs were not found during verification.
8. A dedicated model reproduction script and manifest were not found in the inspected paths.
9. A fresh virtual-environment installation and full pipeline run remain to be completed.
10. Further validation on additional datasets and real operational data would be needed before deployment.

Recommended next steps:
- Confirm the prediction horizon and maintenance cost assumptions with stakeholders.
- Run final evaluation on the reserved test partition using a pre-approved evaluation protocol.
- Complete and verify experiment tracking and model reproducibility.
- Perform a clean-environment installation and end-to-end test.
- Validate alert behaviour with domain experts before production use.

## 12. Conclusion

The Week 4 integration checks confirm that the project loads the FD001 data, passes its 93 automated tests, loads the saved GRU model and preprocessing artifact, and completes streaming inference for 100 test engines.

The engineered Logistic Regression and GRU experiments both show promising validation results. The threshold analysis provides a provisional alert rule, subject to stakeholder approval.

The project is a verified development-stage predictive maintenance prototype. Clean-environment verification, experiment-run evidence, final reserved-test evaluation, and stakeholder approval remain important before describing it as production-ready.
