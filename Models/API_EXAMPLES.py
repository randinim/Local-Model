"""
API Integration Examples
========================
"""

# ============================================================================
# Flask API Example
# ============================================================================

from flask import Flask, request, jsonify
from model_lib import WastePredictor

app = Flask(__name__)
predictor = WastePredictor()  # Load once at startup

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Body: {
        "production_volume": 50000,
        "production_capacity": 60000,
        "rain_sum": 200,
        "temperature_mean": 28,
        "humidity_mean": 85,
        "wind_speed_mean": 15,
        "month": 6,
        "year": 2026
    }
    """
    try:
        data = request.json
        result = predictor.predict_dict(data)
        return jsonify({"success": True, "prediction": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=5000)


# ============================================================================
# FastAPI Example
# ============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_lib import WastePredictor, PredictionInput, PredictionResult

app = FastAPI()
predictor = WastePredictor()

class PredictionRequest(BaseModel):
    production_volume: float
    production_capacity: float
    rain_sum: float
    temperature_mean: float
    humidity_mean: float
    wind_speed_mean: float
    month: int = 6
    year: int = 2026

@app.post("/predict", response_model=dict)
async def predict(req: PredictionRequest):
    """
    POST /predict
    Returns all 14 waste prediction outputs
    """
    try:
        result = predictor.predict(
            production_volume=req.production_volume,
            production_capacity=req.production_capacity,
            rain_sum=req.rain_sum,
            temperature_mean=req.temperature_mean,
            humidity_mean=req.humidity_mean,
            wind_speed_mean=req.wind_speed_mean,
            month=req.month,
            year=req.year
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run with: uvicorn api_examples:app --reload


# ============================================================================
# AWS Lambda Handler
# ============================================================================

from model_lib import WastePredictor
import json

predictor = None

def lambda_handler(event, context):
    """
    AWS Lambda function for waste prediction
    
    Event body should contain:
    {
        "production_volume": 50000,
        "production_capacity": 60000,
        "rain_sum": 200,
        "temperature_mean": 28,
        "humidity_mean": 85,
        "wind_speed_mean": 15,
        "month": 6,
        "year": 2026
    }
    """
    global predictor
    
    # Initialize predictor once (cold start)
    if predictor is None:
        predictor = WastePredictor()
    
    try:
        # Parse input
        if isinstance(event.get('body'), str):
            data = json.loads(event['body'])
        else:
            data = event
        
        # Make prediction
        result = predictor.predict_dict(data)
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': True,
                'prediction': result
            })
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }


# ============================================================================
# Django View Example
# ============================================================================

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from model_lib import WastePredictor
import json

# Initialize once at module level
predictor = WastePredictor()

@csrf_exempt
@require_http_methods(["POST"])
def predict_waste_view(request):
    """
    POST /api/predict-waste/
    """
    try:
        data = json.loads(request.body)
        result = predictor.predict_dict(data)
        return JsonResponse({"success": True, "prediction": result})
    except Exception as e:
        return JsonResponse(
            {"success": False, "error": str(e)},
            status=400
        )


# ============================================================================
# Batch Processing Script
# ============================================================================

import pandas as pd
from model_lib import WastePredictor

def process_batch_file(input_csv, output_csv):
    """
    Process a CSV file with input data and save predictions
    
    Input CSV columns:
        production_volume, production_capacity, rain_sum,
        temperature_mean, humidity_mean, wind_speed_mean,
        Month, Year
    """
    # Load predictor
    predictor = WastePredictor()
    
    # Read input data
    df = pd.read_csv(input_csv)
    
    # Make predictions
    predictions = predictor.predict_batch(df)
    
    # Combine input and predictions
    result = pd.concat([df, predictions], axis=1)
    
    # Save to CSV
    result.to_csv(output_csv, index=False)
    print(f"Processed {len(df)} rows. Saved to {output_csv}")

# Usage:
# process_batch_file('input.csv', 'output_with_predictions.csv')


# ============================================================================
# Real-time Monitoring Script
# ============================================================================

from model_lib import WastePredictor
import time

predictor = WastePredictor()

def monitor_production(production_data_source):
    """
    Continuously monitor production and predict waste
    """
    while True:
        try:
            # Get latest production data (example)
            data = production_data_source.get_latest()
            
            # Make prediction
            result = predictor.predict(**data)
            
            # Alert if waste exceeds threshold
            if result.total_waste_kg > 30000:
                print(f"⚠️  HIGH WASTE ALERT: {result.total_waste_kg:.2f} kg")
            
            # Log results
            print(f"Waste: {result.total_waste_kg:.2f} kg, "
                  f"Bittern: {result.liquid_waste_bittern_liters:.2f} L")
            
            time.sleep(300)  # Check every 5 minutes
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(60)
