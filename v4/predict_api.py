"""
Simple prediction API for v4 model.
Provides `get_waste_prediction(input_dict) -> dict` for programmatic use.
"""
import os
import sys
import importlib.util
import pickle
import pandas as pd
from typing import Dict

# Load train module to make classes available for unpickling
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, 'train.py')
spec = importlib.util.spec_from_file_location('train_v4', train_path)
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
# ensure unpickling can find classes
sys.modules['__main__'] = train_mod

MODEL_PATH = os.path.join(script_dir, 'waste_predictor_v4.pkl')


def _load_model(path=MODEL_PATH):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    # If a ProductionWastePredictor instance was saved, return it
    if hasattr(data, 'predict'):
        return data
    # If a dict was saved, try to reconstruct
    if isinstance(data, dict) and 'models' in data:
        predictor = train_mod.ProductionWastePredictor()
        predictor.models = data['models']
        predictor.weights = data.get('weights', {})
        predictor.feature_engineer.feature_names = data.get('feature_names', [])
        predictor.output_cols = data.get('output_cols', predictor.output_cols)
        predictor.meta_corrector = data.get('meta_corrector', None)
        return predictor
    raise RuntimeError('Unrecognized model format')


def get_waste_prediction(input_data: Dict) -> Dict:
    """Return predictions for a single input dictionary."""
    model = _load_model()
    # Build single-row dataframe
    df = pd.DataFrame([{
        'production_volume': input_data['production_volume'],
        'rain_sum': input_data['rain_sum'],
        'temperature_mean': input_data['temperature_mean'],
        'humidity_mean': input_data['humidity_mean'],
        'wind_speed_mean': input_data['wind_speed_mean'],
        'Month': input_data.get('month', 6),
        'Year': input_data.get('year', 2024)
    }])

    pred_df = model.predict(df)
    return pred_df.iloc[0].to_dict()


if __name__ == '__main__':
    # Example
    sample = {
        'production_volume': 1500000,
        'rain_sum': 250,
        'temperature_mean': 28.5,
        'humidity_mean': 90,
        'wind_speed_mean': 18,
        'month': 7,
        'year': 2024
    }
    print(get_waste_prediction(sample))
