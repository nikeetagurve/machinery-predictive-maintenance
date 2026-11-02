# Experiment Tracking Setup and Verification

- **Task:** Week 3 — Tracking (Add MLflow / W&B experiment tracking)
- **Assignee:** Shankar Savalgi
- **Tool:** MLflow Tracking

## 1. Storage Configuration
- **Tracking URI:** `file://reports/mlruns` (Local, version-controlled filesystem store)
- **Artifacts Directory:** `reports/mlruns/<experiment_id>/<run_id>/artifacts/`
- **Logged Entities:**
  - Parameters: Learning rate, estimators, leaves, depth, subsample, class weighting (`scale_pos_weight`), window size.
  - Metrics: `val_precision`, `val_recall`, `val_f1_score`, `val_roc_auc`.
  - Artifacts: LightGBM model booster (`model/`), confusion matrix (`evaluation/`), feature list (`metadata/`).

## 2. Launching the Dashboard
To inspect logged metrics, run comparisons, and download artifacts:
```bash
mlflow ui --backend-store-uri reports/mlruns --port 5000