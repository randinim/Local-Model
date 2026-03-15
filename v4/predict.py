import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np


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


def predict_csv(model_path, input_csv, output_csv):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    df = pd.read_csv(input_csv, comment='#')
    X = _fe(df)

    scaler = data.get('scaler')
    if scaler is not None:
        Xs = scaler.transform(X)
    else:
        Xs = X.values

    if 'model' in data and hasattr(data['model'], 'predict'):
        preds = data['model'].predict(Xs)
        cols = data.get('output_cols', None)
    elif 'models' in data and isinstance(data['models'], dict):
        models = data['models']
        cols = data.get('output_cols', list(models.keys()))
        preds = np.column_stack([models[c].predict(Xs) for c in cols])
    else:
        raise RuntimeError('Unrecognized model format in pickle')

    out_df = pd.DataFrame(preds, columns=cols, index=df.index)
    out_df.to_csv(output_csv, index=False)
    print('Predictions saved to', output_csv)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(script_dir, 'waste_predictor_v4.pkl')
    p = argparse.ArgumentParser(description='Predict waste outputs with v4 model')
    p.add_argument('--model', default=default_model)
    p.add_argument('--input', required=True, help='Input CSV with features')
    p.add_argument('--output', required=True, help='Output CSV path for predictions')
    args = p.parse_args()

    if not os.path.exists(args.model):
        print('Model not found at', args.model)
        print('Run v4/train.py to produce the model first')
        sys.exit(1)

    predict_csv(args.model, args.input, args.output)
