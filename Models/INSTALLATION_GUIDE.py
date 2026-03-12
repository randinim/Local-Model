"""
INSTALLATION & SETUP GUIDE
===========================

Quick Copy-Paste Commands for Different Scenarios
"""

# ============================================================================
# 1. INSTALL FROM PyPI (when published)
# ============================================================================

# Basic installation
"""
pip install waste-forecasting-model
"""

# With S3 support (for downloading models from cloud)
"""
pip install waste-forecasting-model[sqs]
"""

# With all optional dependencies
"""
pip install waste-forecasting-model[all]
"""


# ============================================================================
# 2. INSTALL FROM LOCAL SOURCE (for development)
# ============================================================================

# Install in editable mode
"""
cd d:\Research\model_lib
pip install -e .
"""

# Or with S3 support
"""
cd d:\Research\model_lib
pip install -e .[sqs]
"""


# ============================================================================
# 3. MODEL FILE SETUP
# ============================================================================

# Option A: Copy model to home directory (recommended)
"""
mkdir -p ~/.model_lib
cp waste_predictor_v1.pkl ~/.model_lib/
cp waste_predictor_v1_metadata.json ~/.model_lib/
"""

# Windows PowerShell:
"""
New-Item -ItemType Directory -Force -Path $env:USERPROFILE\.model_lib
Copy-Item waste_predictor_v1.pkl $env:USERPROFILE\.model_lib\
Copy-Item waste_predictor_v1_metadata.json $env:USERPROFILE\.model_lib\
"""

# Option B: Use environment variable
"""
export WASTE_PREDICTOR_MODEL_PATH="/path/to/waste_predictor_v1.pkl"
"""

# Windows PowerShell:
"""
$env:WASTE_PREDICTOR_MODEL_PATH="D:\path\to\waste_predictor_v1.pkl"
"""

# Option C: Keep in current directory
"""
# Just ensure waste_predictor_v1.pkl is in the same directory as your script
"""


# ============================================================================
# 4. VERIFY INSTALLATION
# ============================================================================

"""
python -c "from model_lib import WastePredictor; p=WastePredictor(); print(f'✓ Model loaded. Version: {p.version}')"
"""


# ============================================================================
# 5. MINIMAL TEST SCRIPT
# ============================================================================

"""
# test_installation.py
from model_lib import predict_waste

result = predict_waste(
    production_volume=50000,
    production_capacity=60000,
    rain_sum=200,
    temperature_mean=28,
    humidity_mean=85,
    wind_speed_mean=15
)

print(f"✓ Prediction successful!")
print(f"Total Waste: {result['Total_Waste_kg']:.2f} kg")
"""


# ============================================================================
# 6. DOCKER SETUP
# ============================================================================

"""
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install package
RUN pip install waste-forecasting-model

# Copy model file
COPY waste_predictor_v1.pkl /root/.model_lib/

# Copy your application
COPY app.py .

CMD ["python", "app.py"]
"""


# ============================================================================
# 7. REQUIREMENTS.TXT (for other projects)
# ============================================================================

"""
# requirements.txt
waste-forecasting-model>=1.0.0
# or from git:
# git+https://github.com/your-org/waste-forecasting-model.git
"""


# ============================================================================
# 8. AWS LAMBDA DEPLOYMENT
# ============================================================================

"""
# 1. Create deployment package
mkdir lambda_package
cd lambda_package
pip install waste-forecasting-model -t .

# 2. Add your model file
mkdir -p .model_lib
cp ../waste_predictor_v1.pkl .model_lib/

# 3. Add your handler
cp ../lambda_handler.py .

# 4. Create ZIP
zip -r function.zip .

# 5. Upload to Lambda via AWS CLI
aws lambda create-function \
  --function-name waste-predictor \
  --runtime python3.9 \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://function.zip \
  --role arn:aws:iam::ACCOUNT-ID:role/lambda-role
"""


# ============================================================================
# 9. GOOGLE COLAB / JUPYTER
# ============================================================================

"""
# Cell 1: Install
!pip install waste-forecasting-model

# Cell 2: Upload model
from google.colab import files
uploaded = files.upload()  # Upload waste_predictor_v1.pkl

# Cell 3: Use
from model_lib import WastePredictor
predictor = WastePredictor(model_path='waste_predictor_v1.pkl')

result = predictor.predict(
    production_volume=50000,
    production_capacity=60000,
    rain_sum=200,
    temperature_mean=28,
    humidity_mean=85,
    wind_speed_mean=15
)
print(result.to_dict())
"""


# ============================================================================
# 10. CI/CD EXAMPLE (GitHub Actions)
# ============================================================================

"""
# .github/workflows/test.yml
name: Test Model Predictions

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install waste-forecasting-model pytest
      
      - name: Download model from artifacts
        run: |
          mkdir -p ~/.model_lib
          # Download from S3 or artifacts
          aws s3 cp s3://my-bucket/waste_predictor_v1.pkl ~/.model_lib/
      
      - name: Run tests
        run: pytest tests/
"""


# ============================================================================
# QUICK REFERENCE - Import Statements
# ============================================================================

# Main predictor class
from model_lib import WastePredictor

# Convenience function
from model_lib import predict_waste

# Type definitions
from model_lib import PredictionInput, PredictionResult

# S3 support
from model_lib import download_model_from_s3

# Constants
from model_lib import OUTPUT_COLUMNS, INPUT_COLUMNS

# Federated Learning (advanced)
from model_lib.fl_adapter import WastePredictorFLAdapter, FLMessage
