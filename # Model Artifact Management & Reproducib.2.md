# Model Artifact Management & Reproducibility Guide

- **Task:** Week 4 — Tracking: Finalize model artifact saving/loading
- **Author:** Shankar Savalgi
- **Module:** `src/modeling/model_io.py` & `scripts/reproduce_model.py`

## 1. Storage & Verification
- Model files are serialized as text boosters (`model.txt`), avoiding pickle dependency issues and keeping file sizes small (< 2 MB).
- The accompanying `manifest.json` stores:
  - Exact required feature list and count.
  - Recommended classification decision threshold.
  - Validation metrics on the holdout split.
  - Cryptographic `SHA256` checksum to verify integrity at load time.

## 2. Reproducibility
To regenerate the exact model artifact from scratch:
```bash
python scripts/reproduce_model.py