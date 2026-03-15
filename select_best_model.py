"""
Select best model using composite score from comparison_results.json.
Composite = 0.5 * norm(R2) + 0.25 * (1 - norm(MAE)) + 0.25 * (1 - norm(RMSE))
Normalization is min-max across available models.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
cmp_path = os.path.join(ROOT, 'comparison_results.json')
with open(cmp_path, 'r') as f:
    data = json.load(f)

# collect numeric entries
rows = {}
for k,v in data.items():
    if isinstance(v, dict) and 'r2' in v:
        rows[k] = {'r2': v['r2'], 'mae': v['mae'], 'rmse': v['rmse']}

if not rows:
    print('No numeric comparison entries found.')
    raise SystemExit(1)

# arrays
r2s = [rows[k]['r2'] for k in rows]
maes = [rows[k]['mae'] for k in rows]
rmses = [rows[k]['rmse'] for k in rows]

r2_min, r2_max = min(r2s), max(r2s)
mae_min, mae_max = min(maes), max(maes)
rm_min, rm_max = min(rmses), max(rmses)

scores = {}
for k, v in rows.items():
    r2 = v['r2']
    mae = v['mae']
    rm = v['rmse']
    # normalize to 0-1 (if all equal, set 0.5)
    if r2_max > r2_min:
        nr2 = (r2 - r2_min) / (r2_max - r2_min)
    else:
        nr2 = 0.5
    if mae_max > mae_min:
        nmae = (mae - mae_min) / (mae_max - mae_min)
    else:
        nmae = 0.5
    if rm_max > rm_min:
        nrm = (rm - rm_min) / (rm_max - rm_min)
    else:
        nrm = 0.5
    # composite: higher better; invert mae and rmse
    composite = 0.5 * nr2 + 0.25 * (1 - nmae) + 0.25 * (1 - nrm)
    scores[k] = {'composite': composite, 'r2': r2, 'mae': mae, 'rmse': rm}

# rank
ranked = sorted(scores.items(), key=lambda x: x[1]['composite'], reverse=True)
print('Model ranking by composite score:')
for name, s in ranked:
    print(f"{name}: composite={s['composite']:.4f}, R2={s['r2']:.4f}, MAE={s['mae']:.2f}, RMSE={s['rmse']:.2f}")

best = ranked[0]
print('\nSelected best model:', best[0])

# Save selection
out = {'ranking': [{name: val} for name, val in ranked], 'selected': best[0]}
with open(os.path.join(ROOT, 'selection_summary.json'), 'w') as f:
    json.dump(out, f, indent=2)
print('Selection saved to selection_summary.json')
