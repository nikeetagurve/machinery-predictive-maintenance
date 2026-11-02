# Model Artifact Management & Reproducibility Guide

- **Date:** November 2, 2026
- **Module:** `src/modeling/model_io.py` & `scripts/reproduce_model.py`
- **Lead:** Shankar Savalgi

---

## 1. Artifact Packaging Format
To prevent pickle incompatibility and ensure lightweight git compliance:
- Model weights are stored in standard text booster format (`model.txt`), keeping file sizes under 2 MB.
- Each artifact bundle contains a companion `manifest.json` detailing:
  - Expected input feature sequence and dimension count.
  - Recommended alert classification threshold.
  - Cryptographic `SHA256` checksum for tampering detection during deployment.
  - Final validation metrics on the holdout split.

## 2. Reproducing the Saved Artifact
To recreate the artifact deterministically from scratch:
```bash
# 1. Clean previous build (if any)
rm -rf models/v1.0.0

# 2. Execute reproducible pipeline
python scripts/reproduce_model.py