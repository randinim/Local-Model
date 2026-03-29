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
    DeepNeuralNetworkModel,
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
        "Total_Waste_kg",
        "Solid_Waste_Gypsum_kg",
        "Solid_Waste_Limestone_kg",
        "Solid_Waste_Industrial_Salt_kg",
        "Total_Solid_Waste_kg",
        "Liquid_Waste_Bittern_Liters",
        "Bittern_Mg_Concentration_gL",
        "Bittern_K_Concentration_gL",
        "Bittern_SO4_Concentration_gL",
        "Bittern_Ca_Concentration_gL",
        "Bittern_Magnesium_kg",
        "Bittern_Potassium_kg",
        "Bittern_Sulfate_kg",
        "Bittern_Calcium_kg",
    ]

    def __init__(self, model_path: str = None):
        """Load the model from pickle file."""
        if model_path is None:
            # Default to same directory as this script
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "waste_predictor_v2.pkl"
            )

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        self.models = data["models"]
        self.weights = data["weights"]
        self.feature_names = data["feature_names"]

    def predict(
        self,
        production_volume: float,
        production_capacity: float,
        rain_sum: float,
        temperature_mean: float,
        humidity_mean: float,
        wind_speed_mean: float,
        month: int = 6,
        year: int = 2024,
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
        input_df = pd.DataFrame(
            [
                {
                    "production_volume": production_volume,
                    "production_capacity": production_capacity,
                    "rain_sum": rain_sum,
                    "temperature_mean": temperature_mean,
                    "humidity_mean": humidity_mean,
                    "wind_speed_mean": wind_speed_mean,
                    "Month": month,
                    "Year": year,
                }
            ]
        )

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
    year: int = 2024,
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
        year=year,
    )

if __name__ == "__main__":
    import csv

    # Define 10 high-value stress scenarios for Puttalam
    stress_scenarios = [
        {"name": "1. Peak Yala Dry (Aug)", "prod": 1800000, "rain": 5, "temp": 33.0, "hum": 60, "wind": 24, "month": 8},
        {"name": "2. Peak Maha Monsoon (Nov)", "prod": 200000, "rain": 550, "temp": 25.5, "hum": 96, "wind": 8, "month": 11},
        {"name": "3. Inter-Monsoon Heat (Mar)", "prod": 1200000, "rain": 120, "temp": 31.0, "hum": 75, "wind": 14, "month": 3},
        {"name": "4. Over-Capacity Run (110%)", "prod": 2200000, "rain": 40, "temp": 30.0, "hum": 70, "wind": 18, "month": 9},
        {"name": "5. Minimum Viable Batch", "prod": 50000, "rain": 20, "temp": 29.5, "hum": 68, "wind": 20, "month": 2},
        {"name": "6. Cyclone/Depression Event", "prod": 300000, "rain": 300, "temp": 24.0, "hum": 98, "wind": 45, "month": 12},
        {"name": "7. Extreme Heatwave", "prod": 1500000, "rain": 0, "temp": 38.0, "hum": 40, "wind": 12, "month": 5},
        {"name": "8. High Humidity/Low Wind", "prod": 900000, "rain": 50, "temp": 28.0, "hum": 92, "wind": 2, "month": 10},
        {"name": "9. Poor Quality Crude Run", "prod": 1400000, "rain": 180, "temp": 27.5, "hum": 85, "wind": 10, "month": 6},
        {"name": "10. High Purity Industrial Run", "prod": 1600000, "rain": 10, "temp": 32.0, "hum": 55, "wind": 28, "month": 7}
    ]

    all_results = []
    
    print("=" * 75)
    print(f"{'SCENARIO NAME':<35} | {'BITTERN (L)':<15} | {'SALT WASTE (kg)':<15}")
    print("-" * 75)

    for s in stress_scenarios:
        # Run prediction
        res = predict_waste(
            production_volume=s['prod'],
            production_capacity=2000000,
            rain_sum=s['rain'],
            temperature_mean=s['temp'],
            humidity_mean=s['hum'],
            wind_speed_mean=s['wind'],
            month=s['month'],
            year=2024
        )
        
        # Log to console
        print(f"{s['name']:<35} | {res['Liquid_Waste_Bittern_Liters']:15,.2f} | {res['Solid_Waste_Industrial_Salt_kg']:15,.2f}")
        
        # Prepare for CSV export
        export_row = {"Scenario": s['name'], **s} # Include inputs
        export_row.update(res) # Include outputs
        all_results.append(export_row)

    # Export to CSV for Excel
    keys = all_results[0].keys()
    with open('Puttalam_10_Scenario_Validation.csv', 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_results)

    print("-" * 75)
    print("\n[SUCCESS] Stress test complete. Results exported to: Puttalam_10_Scenario_Validation.csv")
    print("=" * 75)