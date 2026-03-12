"""
Quick Start Guide for waste-forecasting-model
==============================================

Installation:
    pip install waste-forecasting-model

Or from local source:
    cd d:\\Research\\model_lib
    pip install -e .

Requirements:
    - Python 3.8+
    - Model file: waste_predictor_v1.pkl must be in one of:
        * ~/.model_lib/waste_predictor_v1.pkl
        * ./waste_predictor_v1.pkl (current directory)
        * Set WASTE_PREDICTOR_MODEL_PATH environment variable
"""

# ============================================================================
# Method 1: Simple function call (recommended for one-off predictions)
# ============================================================================

from model_lib import predict_waste

result = predict_waste(
    production_volume=50000,      # kg
    production_capacity=60000,    # kg
    rain_sum=200,                 # mm
    temperature_mean=28,          # °C
    humidity_mean=85,             # %
    wind_speed_mean=15,           # km/h
    month=6,                      # 1-12
    year=2026                     # optional
)

print("Prediction Result:")
print(f"Total Waste: {result['Total_Waste_kg']:.2f} kg")
print(f"Gypsum: {result['Solid_Waste_Gypsum_kg']:.2f} kg")
print(f"Bittern Volume: {result['Liquid_Waste_Bittern_Liters']:.2f} L")
print(f"Mg Concentration: {result['Bittern_Mg_Concentration_gL']:.2f} g/L")


# ============================================================================
# Method 2: Create predictor instance (recommended for multiple predictions)
# ============================================================================

from model_lib import WastePredictor

# Create predictor (loads model once)
predictor = WastePredictor()

# Single prediction with typed result
result = predictor.predict(
    production_volume=50000,
    production_capacity=60000,
    rain_sum=200,
    temperature_mean=28,
    humidity_mean=85,
    wind_speed_mean=15,
    month=6,
    year=2026
)

# Access fields directly
print(f"\nTotal Waste: {result.total_waste_kg:.2f} kg")
print(f"Solid Waste Components:")
print(f"  - Gypsum: {result.solid_waste_gypsum_kg:.2f} kg")
print(f"  - Limestone: {result.solid_waste_limestone_kg:.2f} kg")
print(f"  - Industrial Salt: {result.solid_waste_industrial_salt_kg:.2f} kg")
print(f"Bittern:")
print(f"  - Volume: {result.liquid_waste_bittern_liters:.2f} L")
print(f"  - Mg: {result.bittern_magnesium_kg:.2f} kg")
print(f"  - K: {result.bittern_potassium_kg:.2f} kg")

# Or convert to dict for JSON/API responses
result_dict = result.to_dict()


# ============================================================================
# Method 3: Dictionary input (for API/JSON payloads)
# ============================================================================

input_data = {
    'production_volume': 50000,
    'production_capacity': 60000,
    'rain_sum': 200,
    'temperature_mean': 28,
    'humidity_mean': 85,
    'wind_speed_mean': 15,
    'month': 6,
    'year': 2026
}

result_dict = predictor.predict_dict(input_data)
# Returns plain dict with all 14 outputs


# ============================================================================
# Method 4: Batch predictions (DataFrame)
# ============================================================================

import pandas as pd

# Prepare batch data
batch_data = pd.DataFrame([
    {
        'production_volume': 50000, 'production_capacity': 60000,
        'rain_sum': 200, 'temperature_mean': 28,
        'humidity_mean': 85, 'wind_speed_mean': 15,
        'Month': 6, 'Year': 2026
    },
    {
        'production_volume': 45000, 'production_capacity': 60000,
        'rain_sum': 150, 'temperature_mean': 30,
        'humidity_mean': 75, 'wind_speed_mean': 18,
        'Month': 8, 'Year': 2026
    }
])

# Get predictions for all rows
predictions_df = predictor.predict_batch(batch_data)
# Returns DataFrame with 14 output columns

print("\nBatch Predictions:")
print(predictions_df[['Total_Waste_kg', 'Total_Solid_Waste_kg', 
                      'Liquid_Waste_Bittern_Liters']])


# ============================================================================
# Method 5: Use typed input validation
# ============================================================================

from model_lib import PredictionInput

# Create validated input
validated_input = PredictionInput(
    production_volume=50000,
    production_capacity=60000,
    rain_sum=200,
    temperature_mean=28,
    humidity_mean=85,
    wind_speed_mean=15,
    month=6,
    year=2026
)

# Make prediction with typed input
result = predictor.predict_typed(validated_input)


# ============================================================================
# Method 6: Model metadata
# ============================================================================

print(f"\nModel Version (R²): {predictor.version}")
print(f"Output Columns: {predictor.OUTPUT_COLS}")
print(f"Metadata: {predictor.metadata['test_metrics']}")


# ============================================================================
# Method 7: Download model from S3 (optional - requires boto3)
# ============================================================================

# First time setup:
#   pip install waste-forecasting-model[sqs]

# from model_lib import download_model_from_s3
# 
# # Download to default location
# model_path = download_model_from_s3(
#     "s3://my-bucket/models/waste_predictor_v1.pkl"
# )
# 
# # Use the downloaded model
# predictor = WastePredictor(model_path=str(model_path))

# Or hot-reload existing predictor:
# info = predictor.download_and_update_model(
#     "s3://my-bucket/models/waste_predictor_v1.pkl"
# )
# print(f"Updated to version: {info['version']}")


# ============================================================================
# All 14 Output Fields
# ============================================================================

# result.total_waste_kg                    # Total waste (kg)
# result.solid_waste_gypsum_kg             # Gypsum component (kg)
# result.solid_waste_limestone_kg          # Limestone component (kg)
# result.solid_waste_industrial_salt_kg    # Industrial salt component (kg)
# result.total_solid_waste_kg              # Sum of solid components (kg)
# result.liquid_waste_bittern_liters       # Bittern volume (L)
# result.bittern_mg_concentration_gl       # Mg concentration (g/L)
# result.bittern_k_concentration_gl        # K concentration (g/L)
# result.bittern_so4_concentration_gl      # SO4 concentration (g/L)
# result.bittern_ca_concentration_gl       # Ca concentration (g/L)
# result.bittern_magnesium_kg              # Total Mg mass (kg)
# result.bittern_potassium_kg              # Total K mass (kg)
# result.bittern_sulfate_kg                # Total SO4 mass (kg)
# result.bittern_calcium_kg                # Total Ca mass (kg)
