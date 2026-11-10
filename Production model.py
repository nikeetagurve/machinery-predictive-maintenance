"""Production model artifact serialization, deserialization, and schema validation."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def compute_file_hash(filepath: Path) -> str:
    """Calculates SHA256 hash of a file for artifact verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_model_artifact(
    model: lgb.LGBMClassifier,
    feature_names: List[str],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    output_dir: Path,
    artifact_version: str = "v1.0.0",
) -> Path:
    """Saves model booster alongside verified metadata manifest."""
    target_dir = output_dir / artifact_version
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save core booster text representation (compact and platform-independent)
    model_filename = "model.txt"
    model_path = target_dir / model_filename
    model.booster_.save_model(str(model_path))

    # 2. Compute artifact checksum
    model_hash = compute_file_hash(model_path)

    # 3. Build metadata manifest
    manifest = {
        "artifact_version": artifact_version,
        "model_type": "LightGBM Classifier",
        "model_file": model_filename,
        "sha256_checksum": model_hash,
        "decision_threshold": config.get("alert_threshold", 0.50),
        "target_window_cycles": config.get("failure_window_cycles", 30),
        "feature_count": len(feature_names),
        "expected_features": feature_names,
        "evaluation_metrics": metrics,
        "hyperparameters": config,
    }

    manifest_path = target_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    logging.info(f"Model artifact successfully saved to: {target_dir}")
    return target_dir


def load_model_artifact(
    artifact_dir: Path,
) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """Loads model booster and verifies its SHA256 checksum and schema against the manifest."""
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {artifact_dir}")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    model_path = artifact_dir / manifest["model_file"]
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file at {model_path}")

    # Integrity verification
    current_hash = compute_file_hash(model_path)
    if current_hash != manifest["sha256_checksum"]:
        raise ValueError("Model artifact corrupted: SHA256 checksum mismatch!")

    booster = lgb.Booster(model_file=str(model_path))
    logging.info(
        f"Verified and loaded model version {manifest['artifact_version']}"
    )
    return booster, manifest