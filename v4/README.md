v4 — Waste Predictor (standalone)
===============================

Overview
--------
v4 implements an out-of-fold (OOF) meta-corrector to avoid optimistic validation. The pipeline trains base models (GB, Stacked, NN) on the training split, generates OOF ensemble predictions with K-Fold, fits a Ridge meta-corrector on those OOF predictions to capture inter-target correlations, then re-evaluates on a hold-out validation set.

Files
-----
- `train.py` — full training pipeline (OOF meta-corrector). Produces `waste_predictor_v4.pkl` and metadata.
- `predict.py` — standalone inference script. Usage:

  ```bash
  python v4/predict.py --model v4/waste_predictor_v4.pkl --input path/to/input.csv --output path/to/pred.csv
  ```

- `requirements.txt` — Python dependencies for v4.
- `data/full_dataset.csv` — dataset (copy or symlink from v2).

How v4 differs from v3/v2
------------------------
- v2: baseline ensembles, removed `production_capacity`.
- v3: added a correlation-aware meta-corrector but trained on the validation set (risk of optimistic evaluation).
- v4: trains the meta-corrector on OOF predictions (KFold) within the training set to avoid leakage. This yields honest hold-out evaluation.

How to run
----------
1. Ensure dataset exists at `v4/data/full_dataset.csv` (copy from `v2/data/full_dataset.csv`):

   ```bash
   cp v2/data/full_dataset.csv v4/data/full_dataset.csv
   ```

2. Train:

   ```bash
   python v4/train.py
   ```

3. Predict:

   ```bash
   python v4/predict.py --input v4/data/full_dataset.csv --output v4/predictions.csv
   ```

Notes
-----
- For honest performance estimates for the entire pipeline, consider nested CV or a time-based holdout.
- The repo includes `v2` and `v3` directories for comparison; see `compare_summary.md` in this folder.
