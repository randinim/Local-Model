"""
API Wrapper for Puttalam Waste Predictor
=========================================
Provides a simple function to get waste predictions from input data.
"""

from typing import Dict

# Import model classes for pickle deserialization
from train import (
    AdvancedFeatureEngineer,
    GradientBoostingWasteModel,
    StackedEnsembleModel,
    NeuralNetworkTrainer,
    ProductionWastePredictor,
    DeepNeuralNetworkModel
)

from predict import predict_waste


def get_waste_prediction(input_data: Dict) -> Dict:
    """
    Wrapper function to get waste predictions from input dictionary.

    Args:
        input_data: dict with keys:
            - production_volume (float): Monthly salt production (kg)
            - production_capacity (float): Plant capacity (kg)
            - rain_sum (float): Total monthly rainfall (mm)
            - temperature_mean (float): Average temperature (°C)
            - humidity_mean (float): Average humidity (%)
            - wind_speed_mean (float): Average wind speed (km/h)
            - month (int, optional, default=6): Month (1-12)
            - year (int, optional, default=2024): Year

    Returns:
        dict: Waste prediction results including:
            - Solid waste components (kg)
            - Bittern volume (L)
            - Ion concentrations (g/L)
            - Ion masses (kg)

    Example:
        >>> input_data = {
        ...     'production_volume': 1500000,
        ...     'production_capacity': 2000000,
        ...     'rain_sum': 250,
        ...     'temperature_mean': 28.5,
        ...     'humidity_mean': 90,
        ...     'wind_speed_mean': 18,
        ...     'month': 7,
        ...     'year': 2024
        ... }
        >>> result = get_waste_prediction(input_data)
        >>> print(f"Bittern Mg: {result['Bittern_Mg_Concentration_gL']:.2f} g/L")
    """
    return predict_waste(
        production_volume=input_data['production_volume'],
        production_capacity=input_data['production_capacity'],
        rain_sum=input_data['rain_sum'],
        temperature_mean=input_data['temperature_mean'],
        humidity_mean=input_data['humidity_mean'],
        wind_speed_mean=input_data['wind_speed_mean'],
        month=input_data.get('month', 6),
        year=input_data.get('year', 2024)
    )


if __name__ == '__main__':
    # Example usage
    print("=" * 70)
    print("PUTTALAM WASTE PREDICTION API - EXAMPLE")
    print("=" * 70)

    # Dry season example (high production, high ion concentrations)
    dry_sample = {
        'production_volume': 2000000,
        'production_capacity': 2500000,
        'rain_sum': 50,
        'temperature_mean': 32,
        'humidity_mean': 65,
        'wind_speed_mean': 25,
        'month': 3,  # March (dry season)
        'year': 2024
    }

    print("\n--- DRY SEASON SCENARIO ---")
    print("Input:")
    for k, v in dry_sample.items():
        print(f"  {k}: {v}")

    result_dry = get_waste_prediction(dry_sample)
    print("\nKey Predictions:")
    print(f"  Total Waste: {result_dry['Total_Waste_kg']:,.2f} kg")
    print(f"  Bittern Volume: {result_dry['Liquid_Waste_Bittern_Liters']:,.2f} L")
    print(f"  Mg Concentration: {result_dry['Bittern_Mg_Concentration_gL']:.2f} g/L")
    print(f"  K Concentration: {result_dry['Bittern_K_Concentration_gL']:.2f} g/L")
    print(f"  SO4 Concentration: {result_dry['Bittern_SO4_Concentration_gL']:.2f} g/L")

    # Wet season example (lower production, lower ion concentrations)
    wet_sample = {
        'production_volume': 800000,
        'production_capacity': 2500000,
        'rain_sum': 400,
        'temperature_mean': 26,
        'humidity_mean': 100,
        'wind_speed_mean': 12,
        'month': 11,  # November (wet season)
        'year': 2024
    }

    print("\n--- WET SEASON SCENARIO ---")
    print("Input:")
    for k, v in wet_sample.items():
        print(f"  {k}: {v}")

    result_wet = get_waste_prediction(wet_sample)
    print("\nKey Predictions:")
    print(f"  Total Waste: {result_wet['Total_Waste_kg']:,.2f} kg")
    print(f"  Bittern Volume: {result_wet['Liquid_Waste_Bittern_Liters']:,.2f} L")
    print(f"  Mg Concentration: {result_wet['Bittern_Mg_Concentration_gL']:.2f} g/L")
    print(f"  K Concentration: {result_wet['Bittern_K_Concentration_gL']:.2f} g/L")
    print(f"  SO4 Concentration: {result_wet['Bittern_SO4_Concentration_gL']:.2f} g/L")

    print("\n--- COMPARISON ---")
    print(f"Ion concentrations are {'higher' if result_dry['Bittern_Mg_Concentration_gL'] > result_wet['Bittern_Mg_Concentration_gL'] else 'lower'} in dry season")
    print(f"Dry/Wet Mg ratio: {result_dry['Bittern_Mg_Concentration_gL'] / result_wet['Bittern_Mg_Concentration_gL']:.2f}x")

    print("\n" + "=" * 70)
