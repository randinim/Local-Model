"""
MINIMAL USAGE - Copy & Paste Anywhere
======================================
"""

# Step 1: Install
# pip install waste-forecasting-model

# Step 2: Use it!
from model_lib import predict_waste

# Single prediction
result = predict_waste(
    production_volume=50000,
    production_capacity=60000,
    rain_sum=200,
    temperature_mean=28,
    humidity_mean=85,
    wind_speed_mean=15,
    month=6,
    year=2026
)

print(f"Total Waste: {result['Total_Waste_kg']:.2f} kg")
print(f"Gypsum: {result['Solid_Waste_Gypsum_kg']:.2f} kg")
print(f"Bittern: {result['Liquid_Waste_Bittern_Liters']:.2f} L")
