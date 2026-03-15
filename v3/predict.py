"""
Simple v3 prediction wrapper for the lightweight v3 pickle.

It supports passing either a pandas DataFrame or scalar keyword arguments.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any


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


class WastePredictor:
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'waste_predictor_v3.pkl')
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        # support both legacy dicts and new single-model payload
        self.model = data.get('model') or data.get('models')
        self.scaler = data.get('scaler')
        self.feature_names = data.get('feature_names', [])
        self.output_cols = data.get('output_cols', [])

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _fe(df)
        if self.scaler is not None:
            Xs = self.scaler.transform(X)
        else:
            Xs = X.values

        if hasattr(self.model, 'predict'):
            preds = self.model.predict(Xs)
        else:
            # fallback: if model is a dict of per-target models
            preds = np.column_stack([m.predict(Xs) for m in self.model.values()])

        return pd.DataFrame(preds, columns=self.output_cols, index=df.index)

    def predict(self, **kwargs: Any) -> Dict[str, Any]:
        # allow scalar inputs
        if 'production_volume' in kwargs:
            df = pd.DataFrame({k: [v] for k, v in kwargs.items()})
            out = self.predict_df(df)
            return out.iloc[0].to_dict()
        raise ValueError('Provide either a DataFrame via predict_df or scalar kwargs for a single prediction')
        humidity_mean: float,
        wind_speed_mean: float,
        month: int = 6,
        year: int = 2024
    ) -> Dict[str, float]:
        """
        Predict waste compositions and ion concentrations.

        Args:
            production_volume: Monthly salt production (kg)
            production_capacity: Plant capacity (kg)
            rain_sum: Total monthly rainfall (mm)
            temperature_mean: Average temperature (°C)
            humidity_mean: Average humidity (%)
            wind_speed_mean: Average wind speed (km/h)
            month: Month (1-12), default=6
            year: Year, default=2024

        Returns:
            dict: Waste predictions including:
                - Solid waste components (kg)
                - Bittern volume (liters)
                - Ion concentrations (g/L)
                - Ion masses (kg)
        """
        # Create input dataframe
        input_df = pd.DataFrame([{
            'production_volume': production_volume,
            'production_capacity': production_capacity,
            'rain_sum': rain_sum,
            'temperature_mean': temperature_mean,
            'humidity_mean': humidity_mean,
            'wind_speed_mean': wind_speed_mean,
            'Month': month,
            'Year': year
        }])

        # Make predictions using ensemble
        predictions = []
        weights = []

        for name, model in self.models.items():
            pred = model.predict(input_df).values
            predictions.append(pred)
            weights.append(self.weights[name])

        # Weighted average
        weights_arr = np.array(weights)
        weighted_pred = np.average(predictions, axis=0, weights=weights_arr)

        # Convert to dictionary
        result = {}
        for i, col in enumerate(self.OUTPUT_COLS):
            result[col] = float(weighted_pred[0, i])

        return result


def predict_waste(
    production_volume: float,
    production_capacity: float,
    rain_sum: float,
    temperature_mean: float,
    humidity_mean: float,
    wind_speed_mean: float,
    month: int = 6,
    year: int = 2024
) -> Dict[str, float]:
    """
    Quick prediction function (loads model on each call).

    Args:
        production_volume: Monthly salt production (kg)
        production_capacity: Plant capacity (kg)
        rain_sum: Total monthly rainfall (mm)
        temperature_mean: Average temperature (°C)
        humidity_mean: Average humidity (%)
        wind_speed_mean: Average wind speed (km/h)
        month: Month (1-12), default=6
        year: Year, default=2024

    Returns:
        dict: Waste predictions

    Example:
        >>> result = predict_waste(
        ...     production_volume=50000,
        ...     production_capacity=60000,
        ...     rain_sum=200,
        ...     temperature_mean=28,
        ...     humidity_mean=85,
        ...     wind_speed_mean=15,
        ...     month=6,
        ...     year=2024
        ... )
        >>> print(f"Total Waste: {result['Total_Waste_kg']:.2f} kg")
        >>> print(f"Bittern Mg: {result['Bittern_Mg_Concentration_gL']:.2f} g/L")
    """
    predictor = WastePredictor()
    return predictor.predict(
        production_volume=production_volume,
        production_capacity=production_capacity,
        rain_sum=rain_sum,
        temperature_mean=temperature_mean,
        humidity_mean=humidity_mean,
        wind_speed_mean=wind_speed_mean,
        month=month,
        year=year
    )


if __name__ == '__main__':
    # Example usage
    print("=" * 70)
    print("PUTTALAM WASTE PREDICTION - EXAMPLE")
    print("=" * 70)

    # Test prediction with typical values
    sample = {
        'production_volume': 1500000,
        'production_capacity': 2000000,
        'rain_sum': 250,
        'temperature_mean': 28.5,
        'humidity_mean': 90,
        'wind_speed_mean': 18,
        'month': 7,
        'year': 2024
    }

    print("\nInput:")
    for k, v in sample.items():
        print(f"  {k}: {v}")

    result = predict_waste(**sample)

    print("\nPredictions:")
    print("\nSolid Waste Components:")
    solid_waste_keys = [k for k in result.keys() if 'Solid' in k or k == 'Total_Waste_kg']
    for k in solid_waste_keys:
        print(f"  {k:40s}: {result[k]:>12,.2f} kg")

    print("\nBittern Liquid Waste:")
    print(f"  {'Liquid_Waste_Bittern_Liters':40s}: {result['Liquid_Waste_Bittern_Liters']:>12,.2f} L")

    print("\nIon Concentrations:")
    conc_keys = [k for k in result.keys() if 'Concentration' in k]
    for k in conc_keys:
        ion = k.replace('Bittern_', '').replace('_Concentration_gL', '')
        print(f"  {ion:40s}: {result[k]:>12,.3f} g/L")

    print("\nIon Masses:")
    mass_keys = [k for k in result.keys() if k.startswith('Bittern_') and k.endswith('_kg')]
    for k in mass_keys:
        ion = k.replace('Bittern_', '').replace('_kg', '')
        print(f"  {ion:40s}: {result[k]:>12,.2f} kg")

    print("\n" + "=" * 70)
