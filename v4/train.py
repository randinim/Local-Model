"""
v4: Very simple per-target shallow-tree trainer

This script trains a shallow DecisionTree for each output column. It
is intended as a fast, low-accuracy early prototype that can be used
to emulate early development phases of the v2 model.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler


OUTPUT_COLS = [
    'Total_Waste_kg',
    'Solid_Waste_Gypsum_kg',
    'Solid_Waste_Limestone_kg',
    'Solid_Waste_Industrial_Salt_kg',
    'Total_Solid_Waste_kg',
    'Liquid_Waste_Bittern_Liters',
    'Bittern_Mg_Concentration_gL',
    'Bittern_K_Concentration_gL',
    'Bittern_SO4_Concentration_gL',
    'Bittern_Ca_Concentration_gL',
    'Bittern_Magnesium_kg',
    'Bittern_Potassium_kg',
    'Bittern_Sulfate_kg',
    'Bittern_Calcium_kg'
]


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


def train_and_save(data_path: str = 'data/full_dataset.csv', out_path: str = 'waste_predictor_v4.pkl'):
    df = pd.read_csv(data_path, comment='#')
    X = _fe(df)
    y = df[OUTPUT_COLS].fillna(0)

    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    models = {}
    for i, col in enumerate(OUTPUT_COLS):
        reg = DecisionTreeRegressor(max_depth=4, random_state=42)
        reg.fit(Xs, y[col].values)
        models[col] = reg

    payload = {
        'models': models,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'output_cols': OUTPUT_COLS,
        'role': 'early_model',
        'parent_model': 'v3'
    }

    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)


if __name__ == '__main__':
    data_file = os.path.join(os.path.dirname(__file__), 'data', 'full_dataset.csv')
    out_file = os.path.join(os.path.dirname(__file__), 'waste_predictor_v4.pkl')
    train_and_save(data_file, out_file)
