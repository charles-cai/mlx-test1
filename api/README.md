# MNIST Digit Prediction API

FastAPI service for predicting handwritten digits using a trained CNN model.

## Features

- REST API for digit prediction from uploaded images
- Confidence scores for all 10 digits (0-9)  
- Health checks and model info endpoints
- Interactive API documentation (Swagger UI)

## Quick Start

### Prerequisites

- Python 3.12+
- A trained MNIST model file (`saved_model.pth`)

### Installation

```bash
# Install with uv (recommended)
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

### Running the API

```bash
# Basic usage
python api.py

# Custom options
python api.py --port 8889 --model-path /path/to/model.pth

# Development mode
python api.py --reload
```

**Available at:**
- API: http://localhost:8889
- Docs: http://localhost:8889/docs

## API Endpoints

### POST /predict
Upload an image file to get digit prediction.

**Response**:
```json
{
  "predicted_digit": 7,
  "confidence": 0.9823,
  "confidence_scores": {
    "0": 0.0001, "1": 0.0002, "2": 0.0015,
    "3": 0.0023, "4": 0.0012, "5": 0.0089,
    "6": 0.0004, "7": 0.9823, "8": 0.0028, "9": 0.0003
  },
  "image_hash": "a1b2c3d4e5f6...",
  "filename": "digit.png"
}
```

### Other Endpoints
- `GET /health` - Health check
- `GET /model-info` - Model information  
- `GET /` - API information

## Docker Usage

```bash
# Build and run
docker build -t mnist-api .
docker run -p 8889:8889 mnist-api

# With custom model path
docker run -p 8889:8889 -v /path/to/model:/app/model mnist-api
```

## Usage Examples

### curl
```bash
curl -X POST "http://localhost:8889/predict" \
     -F "file=@digit_image.png"
```

### Python
```python
import requests

with open('digit_image.png', 'rb') as f:
    response = requests.post('http://localhost:8889/predict', files={'file': f})
    result = response.json()
    print(f"Predicted: {result['predicted_digit']}")
```

## Configuration

- `MODEL_PATH` environment variable: Path to model file (default: `../model/saved_model.pth`)
- `--port`: Port to bind to (default: 8889)
- `--host`: Host to bind to (default: 0.0.0.0)
- `--model-path`: Custom model file path
- `--reload`: Development mode with auto-reload

## Image Requirements

- **Format**: PNG, JPG, or JPEG
- **Content**: Handwritten digit (0-9)
- Images are automatically preprocessed (grayscale, 28x28, normalized)