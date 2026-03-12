"""
===========================================
PUTTALAM WASTE PREDICTOR - PREDICTION MODULE
===========================================
Self-contained prediction module for V2 model.

Usage:
    from predict import predict_waste, WastePredictor

    # Quick prediction
    result = predict_waste(
        production_volume=50000,
        production_capacity=60000,
        rain_sum=200,
        temperature_mean=28,
        humidity_mean=85,
        wind_speed_mean=15,
        month=6,
        year=2024
    )
    print(result)
"""

import pickle
import pandas as pd
import numpy as np
import os
from typing import Dict

# Import classes needed for pickle deserialization
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import (
    AdvancedFeatureEngineer,
    GradientBoostingWasteModel,
    StackedEnsembleModel,
    NeuralNetworkTrainer,
    ProductionWastePredictor,
    DeepNeuralNetworkModel
)


class WastePredictor:
    """
    Simple waste predictor for production use.

    Example:
        predictor = WastePredictor()
        result = predictor.predict(
            production_volume=50000,
            production_capacity=60000,
            rain_sum=200,
            temperature_mean=28,
            humidity_mean=85,
            wind_speed_mean=15,
            month=6,
            year=2024
        )
    """

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

    def __init__(self, model_path: str = None):
        """Load the model from pickle file."""
        if model_path is None:
            # Default to same directory as this script
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'puttalam_waste_predictor_v2.pkl'
            )

        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        self.models = data['models']
        self.weights = data['weights']
        self.feature_names = data['feature_names']

    def predict(
        self,
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
