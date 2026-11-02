"""MLflow experiment tracking and model logging pipeline for C-MAPSS degradation prediction."""

import json
import logging
from pathlib import Path
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

EXPERIMENT_NAME = "CMAPSS_Turbofan_RUL_Prediction"
TRACKING_DIR = Path("reports/mlruns")

# Reproducible model configuration
HYPERPARAMS = {
    "model_family": "LightGBM",
    "dataset": "FD001",
    "n_estimators": 250,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 3.0,
    "early_stopping_rounds": 30,
    "random_state": 42,
    "target_window_cycles": 30,
}


def setup_mlflow():
    """Initializes local MLflow tracking backend."""
    TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"file://{TRACKING_DIR.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logging.info(f"MLflow tracking URI: {tracking_uri}")


def load_feature_splits(train_path: Path, val_path: Path):
    """Loads Parquet or CSV feature splits."""
    load_fn = (
        pd.read_parquet if train_path.suffix == ".parquet" else pd.read_csv
    )
    return load_fn(train_path), load_fn(val_path)


def split_features_and_label(df: pd.DataFrame, target_col: str = "label"):
    """Separates identifiers and non-feature columns from input matrix."""
    drop_cols = ["unit_nr", "time_cycles", "RUL", target_col]
    features = [c for c in df.columns if c not in drop_cols]
    return df[features], df[target_col], features


def run_tracked_training(
    train_path: Path, val_path: Path, run_name: str = "lgbm_feature_experiment"
):
    """Executes model training with complete experiment tracking."""
    setup_mlflow()

    train_df, val_df = load_feature_splits(train_path, val_path)
    target_col = "label" if "label" in train_df.columns else "target"

    X_train, y_train, feature_names = split_features_and_label(
        train_df, target_col
    )
    X_val, y_val, _ = split_features_and_label(val_df, target_col)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logging.info(f"Active MLflow Run: {run_id}")

        # 1. Tags
        mlflow.set_tag("developer", "Shankar Savalgi")
        mlflow.set_tag("project_stage", "Week 3 - Tracking")
        mlflow.set_tag("task", "binary_classification_early_alert")

        # 2. Hyperparameters & metadata
        mlflow.log_params(HYPERPARAMS)
        mlflow.log_param("num_features", len(feature_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))

        # 3. Model training
        model = lgb.LGBMClassifier(
            n_estimators=HYPERPARAMS["n_estimators"],
            learning_rate=HYPERPARAMS["learning_rate"],
            num_leaves=HYPERPARAMS["num_leaves"],
            max_depth=HYPERPARAMS["max_depth"],
            subsample=HYPERPARAMS["subsample"],
            colsample_bytree=HYPERPARAMS["colsample_bytree"],
            scale_pos_weight=HYPERPARAMS["scale_pos_weight"],
            random_state=HYPERPARAMS["random_state"],
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=HYPERPARAMS["early_stopping_rounds"],
                    verbose=False,
                )
            ],
        )

        # 4. Evaluation metrics
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)

        precision = float(precision_score(y_val, y_val_pred, zero_division=0))
        recall = float(recall_score(y_val, y_val_pred, zero_division=0))
        f1 = float(f1_score(y_val, y_val_pred, zero_division=0))
        roc_auc = float(roc_auc_score(y_val, y_val_proba))

        mlflow.log_metrics(
            {
                "val_precision": round(precision, 4),
                "val_recall": round(recall, 4),
                "val_f1_score": round(f1, 4),
                "val_roc_auc": round(roc_auc, 4),
            }
        )

        # 5. Save and log artifacts
        cm = confusion_matrix(y_val, y_val_pred).tolist()
        cm_file = Path("temp_confusion_matrix.json")
        with open(cm_file, "w") as f:
            json.dump({"confusion_matrix": cm}, f, indent=2)
        mlflow.log_artifact(str(cm_file), artifact_path="evaluation")
        if cm_file.exists():
            cm_file.unlink()

        features_file = Path("temp_features.json")
        with open(features_file, "w") as f:
            json.dump({"features": feature_names}, f, indent=2)
        mlflow.log_artifact(str(features_file), artifact_path="metadata")
        if features_file.exists():
            features_file.unlink()

        # Log trained model artifact
        mlflow.lightgbm.log_model(model, artifact_path="model")

        logging.info(
            f"Run logged successfully. F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}"
        )
        return run_id


if __name__ == "__main__":
    train_data = Path("data/processed/train_features.parquet")
    val_data = Path("data/processed/val_features.parquet")

    if not train_data.exists():
        train_data = Path("data/processed/train_features.csv")
        val_data = Path("data/processed/val_features.csv")

    if train_data.exists() and val_data.exists():
        run_tracked_training(train_data, val_data)
    else:
        print("Feature splits not detected in data/processed/. Running dry-check.")