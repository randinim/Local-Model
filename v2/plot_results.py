import os
import json
import importlib.util
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set(style="whitegrid")

sns.set(style="whitegrid")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'waste_predictor_v1.pkl')
data_path = os.path.join(script_dir, 'data', 'full_dataset.csv')
plots_dir = os.path.join(script_dir, 'plots')
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'waste_predictor_v1.pkl')
data_path = os.path.join(script_dir, 'data', 'full_dataset.csv')
plots_dir = os.path.join(script_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# Load train module to access ProductionWastePredictor class
spec = importlib.util.spec_from_file_location("train_v2", os.path.join(script_dir, 'train.py'))
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)

# Ensure unpickling finds classes that were defined when `train.py` ran as __main__
import sys
sys.modules['__main__'] = train_mod

# Load model
model = train_mod.ProductionWastePredictor.load(model_path)

# Load data
df = pd.read_csv(data_path, comment='#')

# Make predictions
pred_df = model.predict(df)

# Align columns and ground truth
output_cols = model.output_cols
y_true = df[output_cols].values
y_pred = pred_df[output_cols].values

# Metrics
r2_per = r2_score(y_true, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

# Save metrics summary
metrics = {
    'r2_overall': float(r2_score(y_true, y_pred)),
    'mae': float(mae),
    'rmse': float(rmse),
    'r2_per_target': dict(zip(output_cols, r2_per))
}
with open(os.path.join(plots_dir, 'metrics_summary.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

# 1) Per-target R² bar chart
plt.figure(figsize=(10, 6))
order = np.argsort(r2_per)
sns.barplot(x=np.array(output_cols)[order], y=np.array(r2_per)[order])
plt.xticks(rotation=90)
plt.ylabel('R²')
plt.title('Per-target R² (sorted)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'per_target_r2.png'), dpi=150)
plt.close()

# 2) Parity plots for representative targets (two panels)
representative = ['Total_Waste_kg', 'Bittern_Mg_Concentration_gL']
plt.figure(figsize=(12, 6))
for i, col in enumerate(representative, 1):
    if col not in output_cols:
        continue
    idx = output_cols.index(col)
    plt.subplot(1, 2, i)
    plt.scatter(y_true[:, idx], y_pred[:, idx], alpha=0.6)
    mn = min(y_true[:, idx].min(), y_pred[:, idx].min())
    mx = max(y_true[:, idx].max(), y_pred[:, idx].max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title(f'Parity: {col}')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'parity_representative.png'), dpi=150)
plt.close()

# 3) Residual distribution violin plots across targets (compact)
residuals = y_true - y_pred
res_df = pd.DataFrame(residuals, columns=output_cols)
ordered_cols = list(np.array(output_cols)[order])
plt.figure(figsize=(12, 6))
sns.violinplot(data=res_df[ordered_cols])
plt.xticks(range(len(ordered_cols)), ordered_cols, rotation=90)
plt.title('Residual distributions (sorted by R²)')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'residuals_violin.png'), dpi=150)
plt.close()

# 4) Feature importance (from gradient boosting component if available)
fi_agg = None
fi_features = None
try:
    gb_container = model.models.get('gradient_boosting')
    if gb_container is not None:
        # gb_container.models maps target -> estimator
        fi_list = []
        for tgt, est in gb_container.models.items():
            if hasattr(est, 'feature_importances_'):
                fi_list.append(est.feature_importances_)
        if fi_list:
            fi_arr = np.vstack(fi_list)
            fi_mean = fi_arr.mean(axis=0)
            fi_agg = fi_mean
            fi_features = gb_container.feature_engineer.feature_names
except Exception:
    fi_agg = None

if fi_agg is not None and fi_features is not None:
    # Top 15 features
    idx_sort = np.argsort(fi_agg)[::-1]
    topk = 15
    top_idx = idx_sort[:topk]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fi_agg[top_idx], y=np.array(fi_features)[top_idx])
    plt.xlabel('Mean feature importance')
    plt.title('Top feature importances (GB average across targets)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance_top15.png'), dpi=150)
    plt.close()

# 5) Error vs Production Volume (heteroscedasticity check) for Total_Waste_kg
if 'Total_Waste_kg' in output_cols:
    idx = output_cols.index('Total_Waste_kg')
    prod = df['production_volume'] if 'production_volume' in df.columns else None
    if prod is not None:
        errs = (y_true[:, idx] - y_pred[:, idx])
        plt.figure(figsize=(6, 4))
        plt.scatter(prod, errs, alpha=0.6)
        sns.regplot(x=prod, y=errs, scatter=False, lowess=True, color='r')
        plt.xlabel('Production volume')
        plt.ylabel('Residual (True - Pred)')
        plt.title('Residual vs Production Volume (Total_Waste_kg)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'error_vs_production.png'), dpi=150)
        plt.close()

# 6) Time-series: True vs Predicted for Total_Waste_kg (sorted by date)
if 'Year' in df.columns and 'Month' in df.columns and 'Total_Waste_kg' in output_cols:
    df_ts = df[['Year', 'Month', 'production_volume']].copy()
    df_ts['date'] = pd.to_datetime(df['Year'].astype(int).astype(str) + '-' + df['Month'].astype(int).astype(str) + '-01')
    df_ts = df_ts.sort_values('date')
    idx = output_cols.index('Total_Waste_kg')
    true_ts = y_true[:, idx]
    pred_ts = y_pred[:, idx]
    plt.figure(figsize=(12, 4))
    plt.plot(df_ts['date'], true_ts[df_ts.index], label='True')
    plt.plot(df_ts['date'], pred_ts[df_ts.index], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Total_Waste_kg')
    plt.title('Time series: True vs Predicted (Total_Waste_kg)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'timeseries_total_waste.png'), dpi=150)
    plt.close()

print('Plots saved to:', plots_dir)
print('Metrics saved to:', os.path.join(plots_dir, 'metrics_summary.json'))
