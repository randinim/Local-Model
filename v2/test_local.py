"""
Test Script for Puttalam Waste Predictor V2
============================================
Test the trained model with various scenarios.
"""

import pandas as pd
import numpy as np
from predict import WastePredictor, predict_waste
from predict_api import get_waste_prediction


def test_single_prediction():
    """Test a single prediction"""
    print("=" * 70)
    print("TEST 1: SINGLE PREDICTION")
    print("=" * 70)

    result = predict_waste(
        production_volume=1500000,
        production_capacity=2000000,
        rain_sum=250,
        temperature_mean=28.5,
        humidity_mean=90,
        wind_speed_mean=18,
        month=7,
        year=2024
    )

    print("\nPrediction Results:")
    for key, value in result.items():
        if 'Concentration' in key:
            print(f"  {key:40s}: {value:>10.3f} g/L")
        else:
            print(f"  {key:40s}: {value:>10,.2f} kg/L")

    # Verify consistency: Total_Solid_Waste_kg = sum of components
    solid_sum = (result['Solid_Waste_Gypsum_kg'] + 
                 result['Solid_Waste_Limestone_kg'] + 
                 result['Solid_Waste_Industrial_Salt_kg'])
    
    print(f"\nConsistency Check:")
    print(f"  Sum of solid components: {solid_sum:,.2f} kg")
    print(f"  Total_Solid_Waste_kg: {result['Total_Solid_Waste_kg']:,.2f} kg")
    print(f"  Difference: {abs(solid_sum - result['Total_Solid_Waste_kg']):,.2f} kg")

    # Verify ion mass = volume * concentration / 1000
    for ion, conc_key, mass_key in [
        ('Mg', 'Bittern_Mg_Concentration_gL', 'Bittern_Magnesium_kg'),
        ('K', 'Bittern_K_Concentration_gL', 'Bittern_Potassium_kg'),
        ('SO4', 'Bittern_SO4_Concentration_gL', 'Bittern_Sulfate_kg'),
        ('Ca', 'Bittern_Ca_Concentration_gL', 'Bittern_Calcium_kg')
    ]:
        calculated_mass = result['Liquid_Waste_Bittern_Liters'] * result[conc_key] / 1000
        print(f"  {ion} mass consistency: {result[mass_key]:,.2f} kg (calculated: {calculated_mass:,.2f} kg)")


def test_seasonal_variation():
    """Test predictions across different seasons"""
    print("\n" + "=" * 70)
    print("TEST 2: SEASONAL VARIATION")
    print("=" * 70)

    scenarios = [
        {
            'name': 'Dry Season (March)',
            'production_volume': 2000000,
            'production_capacity': 2500000,
            'rain_sum': 50,
            'temperature_mean': 32,
            'humidity_mean': 65,
            'wind_speed_mean': 25,
            'month': 3
        },
        {
            'name': 'Monsoon (November)',
            'production_volume': 800000,
            'production_capacity': 2500000,
            'rain_sum': 400,
            'temperature_mean': 26,
            'humidity_mean': 100,
            'wind_speed_mean': 12,
            'month': 11
        },
        {
            'name': 'Peak Production (July)',
            'production_volume': 2200000,
            'production_capacity': 2500000,
            'rain_sum': 80,
            'temperature_mean': 31,
            'humidity_mean': 70,
            'wind_speed_mean': 22,
            'month': 7
        }
    ]

    results = []
    for scenario in scenarios:
        name = scenario.pop('name')
        result = predict_waste(**scenario, year=2024)
        results.append({'name': name, 'result': result, 'input': scenario})

    # Display comparison
    print("\nComparison Across Seasons:")
    print(f"{'Scenario':<25} {'Production':<15} {'Bittern (L)':<15} {'Mg (g/L)':<12} {'K (g/L)':<12}")
    print("-" * 80)
    
    for item in results:
        name = item['name']
        inp = item['input']
        res = item['result']
        print(f"{name:<25} {inp['production_volume']:>13,.0f}  {res['Liquid_Waste_Bittern_Liters']:>13,.0f}  "
              f"{res['Bittern_Mg_Concentration_gL']:>10.2f}  {res['Bittern_K_Concentration_gL']:>10.2f}")


def test_weather_impact():
    """Test impact of weather on ion concentrations"""
    print("\n" + "=" * 70)
    print("TEST 3: WEATHER IMPACT ON ION CONCENTRATIONS")
    print("=" * 70)

    base_params = {
        'production_volume': 1500000,
        'production_capacity': 2000000,
        'month': 6,
        'year': 2024
    }

    weather_scenarios = [
        {
            'name': 'Hot & Dry (High Evap)',
            'rain_sum': 20,
            'temperature_mean': 35,
            'humidity_mean': 50,
            'wind_speed_mean': 30
        },
        {
            'name': 'Moderate',
            'rain_sum': 150,
            'temperature_mean': 28,
            'humidity_mean': 80,
            'wind_speed_mean': 18
        },
        {
            'name': 'Wet & Cool (Low Evap)',
            'rain_sum': 500,
            'temperature_mean': 24,
            'humidity_mean': 100,
            'wind_speed_mean': 10
        }
    ]

    print("\nExpected: Hot/Dry → Higher Ion Concentrations")
    print("          Wet/Cool → Lower Ion Concentrations\n")
    
    print(f"{'Weather Condition':<25} {'Mg (g/L)':<12} {'K (g/L)':<12} {'SO4 (g/L)':<12}")
    print("-" * 65)

    for scenario in weather_scenarios:
        name = scenario.pop('name')
        params = {**base_params, **scenario}
        result = predict_waste(**params)
        
        print(f"{name:<25} {result['Bittern_Mg_Concentration_gL']:>10.2f}  "
              f"{result['Bittern_K_Concentration_gL']:>10.2f}  "
              f"{result['Bittern_SO4_Concentration_gL']:>10.2f}")


def test_capacity_utilization():
    """Test impact of capacity utilization"""
    print("\n" + "=" * 70)
    print("TEST 4: CAPACITY UTILIZATION IMPACT")
    print("=" * 70)

    capacity = 2500000
    weather = {
        'rain_sum': 200,
        'temperature_mean': 29,
        'humidity_mean': 85,
        'wind_speed_mean': 18,
        'month': 6,
        'year': 2024
    }

    utilization_levels = [0.3, 0.5, 0.7, 0.9]
    
    print(f"\n{'Utilization':<15} {'Production (kg)':<18} {'Total Waste (kg)':<18} {'Bittern (L)':<15}")
    print("-" * 70)

    for util in utilization_levels:
        production = capacity * util
        result = predict_waste(
            production_volume=production,
            production_capacity=capacity,
            **weather
        )
        
        print(f"{util*100:>6.0f}%          {production:>15,.0f}  {result['Total_Waste_kg']:>15,.0f}  "
              f"{result['Liquid_Waste_Bittern_Liters']:>13,.0f}")


def test_batch_predictions():
    """Test batch predictions from CSV"""
    print("\n" + "=" * 70)
    print("TEST 5: BATCH PREDICTIONS FROM DATA FILE")
    print("=" * 70)

    # Load a few samples from the training data
    data_path = 'data/full_dataset.csv'
    df = pd.read_csv(data_path, comment='#')
    
    # Take 5 random samples
    samples = df.sample(5, random_state=42)
    
    predictor = WastePredictor()
    
    print(f"\nPredicting {len(samples)} random samples...")
    print(f"{'Sample':<10} {'Actual Waste':<15} {'Pred Waste':<15} {'Diff %':<10}")
    print("-" * 55)
    
    for idx, row in samples.iterrows():
        result = predictor.predict(
            production_volume=row['production_volume'],
            production_capacity=row['production_capacity'],
            rain_sum=row['rain_sum'],
            temperature_mean=row['temperature_mean'],
            humidity_mean=row['humidity_mean'],
            wind_speed_mean=row['wind_speed_mean'],
            month=row['Month'],
            year=row['Year']
        )
        
        actual = row['Total_Waste_kg']
        predicted = result['Total_Waste_kg']
        diff_pct = abs(actual - predicted) / actual * 100
        
        print(f"{idx:<10} {actual:>13,.0f}  {predicted:>13,.0f}  {diff_pct:>8.2f}%")


def test_concentration_ranges():
    """Verify ion concentrations are in expected ranges"""
    print("\n" + "=" * 70)
    print("TEST 6: ION CONCENTRATION RANGE VALIDATION")
    print("=" * 70)

    # Expected ranges from dataset description
    expected_ranges = {
        'Bittern_Mg_Concentration_gL': (6.35, 18.42),
        'Bittern_K_Concentration_gL': (1.53, 4.43),
        'Bittern_SO4_Concentration_gL': (8.65, 25.09),
        'Bittern_Ca_Concentration_gL': (0.03, 0.11)
    }

    # Test with extreme weather conditions
    scenarios = [
        {
            'name': 'Extreme Dry',
            'production_volume': 2500000,
            'production_capacity': 2800000,
            'rain_sum': 10,
            'temperature_mean': 36,
            'humidity_mean': 40,
            'wind_speed_mean': 35,
            'month': 4
        },
        {
            'name': 'Extreme Wet',
            'production_volume': 600000,
            'production_capacity': 2800000,
            'rain_sum': 600,
            'temperature_mean': 23,
            'humidity_mean': 100,
            'wind_speed_mean': 8,
            'month': 12
        }
    ]

    print(f"\n{'Ion':<10} {'Expected Range':<20} {'Dry Pred':<12} {'Wet Pred':<12} {'Status':<10}")
    print("-" * 70)

    results_scenarios = [predict_waste(**s, year=2024) for s in scenarios]
    
    for ion_key, (min_val, max_val) in expected_ranges.items():
        ion_name = ion_key.replace('Bittern_', '').replace('_Concentration_gL', '')
        dry_val = results_scenarios[0][ion_key]
        wet_val = results_scenarios[1][ion_key]
        
        # Check if in range
        dry_ok = min_val <= dry_val <= max_val
        wet_ok = min_val <= wet_val <= max_val
        
        status = "✓ OK" if (dry_ok and wet_ok) else "⚠ CHECK"
        
        print(f"{ion_name:<10} ({min_val:.2f}-{max_val:.2f}){'':<7} {dry_val:>10.2f}  "
              f"{wet_val:>10.2f}  {status:<10}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("PUTTALAM WASTE PREDICTOR V2 - TEST SUITE")
    print("=" * 70)

    try:
        test_single_prediction()
        test_seasonal_variation()
        test_weather_impact()
        test_capacity_utilization()
        test_batch_predictions()
        test_concentration_ranges()
        
        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n⚠ Error: {e}")
        print("Please train the model first by running: python train.py")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()
