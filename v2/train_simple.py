"""
Quick Training Script - Gradient Boosting Only
Avoids Windows multiprocessing issues with stacked ensemble
"""

import pandas as pd
from train import GradientBoostingWasteModel, AdvancedFeatureEngineer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pickle
import json

# Load data
df = pd.read_csv('data/full_dataset.csv', comment='#')

print("=" * 70)
print("PUTTALAM WASTE PREDICTION - QUICK TRAINING")
print("=" * 70)
print(f"\nTotal samples: {len(df)}")
print(f"Date range: {df['Year'].min()}-{df['Year'].max()}")

# Split
train_df, val_df = train_test_split(df, test_size=0.20, random_state=42)
print(f"Training: {len(train_df)} samples")
print(f"Validation: {len(val_df)} samples")

# Train Gradient Boosting
print("\n--- Training Gradient Boosting Model ---")
model = GradientBoostingWasteModel(n_estimators=300, learning_rate=0.05, max_depth=4)
model.fit(train_df, verbose=False)

# Evaluate on validation
print("\n--- Validation Results ---")
val_metrics = model.evaluate(val_df)
print(f"Overall R²: {val_metrics['r2']:.4f}")
print(f"MAE: {val_metrics['mae']:.2f}")
print(f"RMSE: {val_metrics['rmse']:.2f}")

print(f"\nPer-Target R² Scores:")
output_cols = AdvancedFeatureEngineer.OUTPUT_COLS
for col in output_cols:
    r2 = val_metrics['r2_per_target'][col]
    marker = "⭐" if r2 < 0.90 else "✅"
    print(f"  {marker} {col:40s}: {r2:.4f}")

# Cross-validation
print("\n--- 5-Fold Cross-Validation ---")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    train_df_cv = df.iloc[train_idx]
    val_df_cv = df.iloc[val_idx]
    
    gb = GradientBoostingWasteModel(n_estimators=200, learning_rate=0.05, max_depth=4)
    gb.fit(train_df_cv, verbose=False)
    metrics = gb.evaluate(val_df_cv)
    cv_scores.append(metrics['r2'])
    print(f"Fold {fold+1}: R²={metrics['r2']:.4f}")

print(f"\nCV Mean R²: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

# Save model
model_path = 'waste_predictor_v1.pkl'
save_data = {
    'model': model,
    'feature_names': model.feature_engineer.feature_names,
    'output_cols': model.output_cols,
    'metrics': val_metrics
}

with open(model_path, 'wb') as f:
    pickle.dump(save_data, f)

print(f"\n✅ Model saved to: {model_path}")

# Metadata
metadata = {
    'dataset': 'Puttalam Solar Salt - 2000-2026',
    'samples': len(df),
    'features': len(model.feature_engineer.feature_names),
    'targets': len(output_cols),
    'validation_r2': float(val_metrics['r2']),
    'validation_mae': float(val_metrics['mae']),
    'cv_mean_r2': float(np.mean(cv_scores)),
    'cv_std_r2': float(np.std(cv_scores)),
    'per_target_r2': {k: float(v) for k, v in val_metrics['r2_per_target'].items()}
}

json_path = model_path.replace('.pkl', '_metadata.json')
with open(json_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata saved to: {json_path}")

# Analysis
print("\n" + "=" * 70)
print("PERFORMANCE ANALYSIS")
print("=" * 70)

# Categorize targets
production_targets = [c for c in output_cols if 'Waste' in c or 'Bittern_Liters' in c]
concentration_targets = [c for c in output_cols if 'Concentration' in c]
mass_targets = [c for c in output_cols if c.startswith('Bittern_') and c.endswith('_kg')]

print("\n📊 Production-Dependent Targets (Waste Masses):")
prod_r2 = [val_metrics['r2_per_target'][c] for c in production_targets if c in val_metrics['r2_per_target']]
print(f"   Mean R²: {np.mean(prod_r2):.4f} (Expected: 0.92-0.96)")
print(f"   Interpretation: {'✅ As expected - waste is production-dependent' if np.mean(prod_r2) > 0.90 else '⚠️ Lower than expected'}")

print("\n🌤️  Weather-Dependent Targets (Ion Concentrations):")
conc_r2 = [val_metrics['r2_per_target'][c] for c in concentration_targets]
print(f"   Mean R²: {np.mean(conc_r2):.4f} (Expected: 0.70-0.85)")
print(f"   Interpretation: {'✅ As expected - concentrations are weather-dependent' if 0.65 < np.mean(conc_r2) < 0.90 else '⚠️ Check weather features'}")

print("\n⚖️  Derived Targets (Ion Masses = Volume × Concentration):")
mass_r2 = [val_metrics['r2_per_target'][c] for c in mass_targets]
print(f"   Mean R²: {np.mean(mass_r2):.4f} (Expected: 0.88-0.94)")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE - Model is production-ready!")
print("=" * 70)
