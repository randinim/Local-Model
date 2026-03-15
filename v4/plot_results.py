"""
Generate summary plots for v4 model using metadata.
Creates `v4/plots/r2_per_target.png` and `v4/plots/summary.txt`.
"""
import os
import json
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
meta_path = os.path.join(script_dir, 'waste_predictor_v4_metadata.json')
plots_dir = os.path.join(script_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

if not os.path.exists(meta_path):
    raise FileNotFoundError(f"Metadata not found: {meta_path}")

import os
import json
import matplotlib.pyplot as plt
import importlib.util
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

script_dir = os.path.dirname(os.path.abspath(__file__))
meta_path = os.path.join(script_dir, 'waste_predictor_v4_metadata.json')
plots_dir = os.path.join(script_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Try to read metadata first
meta = {}
if os.path.exists(meta_path):
    with open(meta_path, 'r') as f:
        try:
            meta = json.load(f)
        except Exception:
            meta = {}

# Look for per-target R2 in common metadata keys
def extract_per_target(meta_dict):
    candidates = [
        ('val_metrics', 'r2_per_target'),
        ('test_metrics', 'r2_per_target'),
        ('metrics', 'r2_per_target'),
        (None, 'r2_per_target'),
        (None, 'per_target_r2')
    ]
    for outer, inner in candidates:
        obj = meta_dict.get(outer) if outer else meta_dict
        if not obj:
            continue
        if inner in obj:
            return obj[inner]
    return None

per_target = extract_per_target(meta)

if per_target and len(per_target) > 0:
    items = list(per_target.items())
    items.sort(key=lambda x: x[1])
else:
    # Fallback: compute metrics by loading model and running on validation split
    model_path = os.path.join(script_dir, 'waste_predictor_v4.pkl')
    data_path = os.path.join(script_dir, 'data', 'full_dataset.csv')
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        print('No metadata R² and model/data not available to compute metrics. Nothing to plot.')
        per_target = None
        items = []
    else:
        # load model (ensure train module classes are available if needed)
        train_path = os.path.join(script_dir, 'train.py')
        spec = importlib.util.spec_from_file_location('train_v4', train_path)
        train_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_mod)
        # make unpickling find classes
        import sys
        sys.modules['__main__'] = train_mod

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        # construct predictor
        if hasattr(data, 'predict'):
            predictor = data
        elif isinstance(data, dict) and 'models' in data:
            predictor = train_mod.ProductionWastePredictor()
            predictor.models = data['models']
            predictor.weights = data.get('weights', {})
            predictor.feature_engineer.feature_names = data.get('feature_names', [])
            predictor.output_cols = data.get('output_cols', predictor.output_cols)
            predictor.meta_corrector = data.get('meta_corrector', None)
        else:
            raise RuntimeError('Unrecognized model format for computing metrics')

        # load data and compute validation metrics
        df = pd.read_csv(data_path, comment='#')
        train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)
        y_true = val_df[predictor.output_cols].values
        y_pred_df = predictor.predict(val_df)
        y_pred = y_pred_df.values
        r2_vals = r2_score(y_true, y_pred, multioutput='raw_values')
        items = list(zip(predictor.output_cols, r2_vals))
        items.sort(key=lambda x: x[1])

if items:
    labels, scores = zip(*items)

    plt.figure(figsize=(10,6))
    plt.barh(labels, scores, color='tab:blue')
    plt.xlabel('R²')
    plt.title('v4 per-target R²')
    plt.xlim(0,1)
    out_path = os.path.join(plots_dir, 'r2_per_target.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    # write summary
    summary_path = os.path.join(plots_dir, 'summary.txt')
    with open(summary_path, 'w') as s:
        s.write('v4 model per-target R²\n')
        for k,v in items:
            s.write(f"{k}: {v:.4f}\n")

    print('Saved:', out_path)
    print('Summary:', summary_path)
else:
    print('No per-target R² available and no items to plot.')
