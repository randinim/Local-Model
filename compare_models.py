import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set(style='whitegrid')

ROOT = os.path.dirname(__file__)
DATA = os.path.join(ROOT, 'v2', 'data', 'full_dataset.csv')
MODELS = ['v2', 'v3', 'v4']
OUTPUT_DIR = os.path.join(ROOT, 'comparison_plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _fe(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame()
    f['production_volume'] = df['production_volume']
    f['rain_sum'] = df.get('rain_sum', pd.Series(0, index=df.index))
    f['temperature_mean'] = df.get('temperature_mean', pd.Series(0, index=df.index))
    f['humidity_mean'] = df.get('humidity_mean', pd.Series(0, index=df.index))
    f['wind_speed_mean'] = df.get('wind_speed_mean', pd.Series(0, index=df.index))
    if 'Month' in df.columns:
        f['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        f['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    return f.fillna(0)


def load_model_pickle(path, model_dir=None):
    # If the model's train module exists, load it so unpickling can find classes
    if model_dir:
        train_py = os.path.join(model_dir, 'train.py')
        if os.path.exists(train_py):
            import importlib.util, sys
            spec = importlib.util.spec_from_file_location(f"train_{os.path.basename(model_dir)}", train_py)
            train_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(train_mod)
            sys.modules['__main__'] = train_mod

    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_with_payload(payload, df: pd.DataFrame):
    X = _fe(df)
    scaler = payload.get('scaler')
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X.values

    # Multi-output single estimator
    if 'model' in payload and hasattr(payload['model'], 'predict'):
        preds = payload['model'].predict(Xs)
        cols = payload.get('output_cols')
        return pd.DataFrame(preds, columns=cols, index=df.index)

    # Dict of models: could be per-target regressors OR ensemble members
    if 'models' in payload and isinstance(payload['models'], dict):
        models = payload['models']
        # Try to call each model; handle both sklearn-like predict(X) and custom predict(df)
        member_preds = {}
        for name, m in models.items():
            try:
                out = m.predict(Xs)
            except Exception:
                out = m.predict(df)

            # Normalize to DataFrame
            if isinstance(out, pd.DataFrame):
                member_preds[name] = out
            else:
                # numpy array
                arr = np.asarray(out)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                ncols = arr.shape[1]
                # Choose column names: prefer payload output_cols when sizes match
                cols = None
                if payload.get('output_cols') and len(payload.get('output_cols')) == ncols:
                    cols = payload.get('output_cols')
                elif ncols == 1:
                    cols = [name]
                else:
                    cols = [f'col_{i}' for i in range(ncols)]
                member_preds[name] = pd.DataFrame(arr, columns=cols, index=df.index)

        # If members are per-target (one column each) then stack per-output
        # Detect if each member produced single-column predictions
        first = next(iter(member_preds.values()))
        if first.shape[1] == 1:
            cols = payload.get('output_cols', list(member_preds.keys()))
            stacked = np.column_stack([member_preds[c].iloc[:, 0].values for c in cols])
            return pd.DataFrame(stacked, columns=cols, index=df.index)

        # Otherwise treat members as full-output predictors and combine by weights
        weights = payload.get('weights') or {}
        # default equal weights
        w = {name: float(weights.get(name, 1.0)) for name in member_preds.keys()}
        total_w = sum(w.values()) or 1.0

        combined = None
        for name, df_pred in member_preds.items():
            weight = w.get(name, 1.0) / total_w
            vals = df_pred.values * weight
            combined = vals if combined is None else combined + vals

        cols = payload.get('output_cols', first.columns.tolist())
        return pd.DataFrame(combined, columns=cols, index=df.index)

    # Legacy: object with predict method
    if hasattr(payload, 'predict'):
        return payload.predict(df)

    raise RuntimeError('Unrecognized model payload format')


def main():
    df = pd.read_csv(DATA, comment='#')
    results = {}

    for m in MODELS:
        pth = os.path.join(ROOT, m, f'waste_predictor_{m}.pkl')
        if not os.path.exists(pth):
            print(f'Skipping {m}: model pickle not found at {pth}')
            continue

        payload = load_model_pickle(pth, model_dir=os.path.join(ROOT, m))
        preds = predict_with_payload(payload, df)

        # Align output columns and compute metrics
        if hasattr(preds, 'columns'):
            output_cols = list(preds.columns)
        else:
            # fallback to v2 metadata ordering
            from v2 import waste_predictor_v2_metadata as md
            output_cols = md['output_columns']

        y_true = df[output_cols].values
        y_pred = preds[output_cols].values

        r2_per = r2_score(y_true, y_pred, multioutput='raw_values')
        overall = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        results[m] = {
            'overall_r2': overall,
            'mae': mae,
            'rmse': rmse,
            'r2_per_target': dict(zip(output_cols, r2_per))
        }

    # Save metrics
    with open(os.path.join(OUTPUT_DIR, 'compare_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot per-target R2 across models (heatmap)
    # Build matrix
    all_cols = []
    for m, info in results.items():
        for c in info['r2_per_target'].keys():
            if c not in all_cols:
                all_cols.append(c)

    mat = np.zeros((len(results), len(all_cols)))
    models = list(results.keys())
    for i, m in enumerate(models):
        for j, c in enumerate(all_cols):
            mat[i, j] = results[m]['r2_per_target'].get(c, np.nan)

    plt.figure(figsize=(12, 3 + len(models) * 0.5))
    sns.heatmap(mat, xticklabels=all_cols, yticklabels=models, annot=True, fmt='.3f', cmap='viridis')
    plt.xticks(rotation=90)
    plt.title('Per-target R² comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'per_target_r2_comparison.png'), dpi=150)

    # Parity plots for two representative targets
    rep = ['Total_Waste_kg', 'Bittern_Mg_Concentration_gL']
    for target in rep:
        plt.figure(figsize=(8, 6))
        for m in models:
            p = None
            try:
                p = load_model_pickle(os.path.join(ROOT, m, f'waste_predictor_{m}.pkl'), model_dir=os.path.join(ROOT, m))
                preds = predict_with_payload(p, df)
                plt.scatter(df[target], preds[target], alpha=0.5, label=m)
            except Exception:
                continue
        mn = df[rep[0]].min()
        mx = df[rep[0]].max()
        plt.plot([mn, mx], [mn, mx], 'k--')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.title(f'Parity: {target}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'parity_{target}.png'), dpi=150)

    print('Comparison complete. Outputs in', OUTPUT_DIR)


if __name__ == '__main__':
    main()
