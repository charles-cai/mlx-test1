# MNIST Model Package

Complete MNIST digit recognition system with training, debugging, and API deployment.

## 🚀 Quick Start

### Train a Model
```bash
cd model
python MnistModel.py --train --epochs 5
```

### Run the API
```bash
cd api
python api.py
# Open http://localhost:8000
```

### Use Docker
```bash
docker build -t mnist-api .
docker run -p 8000:8000 mnist-api
```

## 📁 File Structure

```
model/
├── MnistModel.py      # Main model class (training + prediction)
├── debug_model.py     # Enhanced debugging wrapper
├── debug_utils.py     # Visualization utilities  
├── train_with_logging.py  # TensorBoard/W&B training
└── README.md          # This file

api/
├── api.py            # FastAPI web service
├── index.html        # Web interface
└── README.md         # API documentation
```

## 🎯 Core Usage

### Basic Training
```python
from MnistModel import MnistModel

model = MnistModel()
train_loader, test_loader = model.get_datasets()

# Train
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
model.train_model(train_loader, optimizer, criterion, num_epochs=5)

# Save
model.save_model("my_model.pth")
```

### Prediction
```python
# Load and predict
model = MnistModel.load_trained_model("saved_model.pth")

with open("digit_image.png", "rb") as f:
    result = model.predict_digit_from_image(f.read())
    print(f"Predicted: {result['predicted_digit']}")
```

## 🔍 Debugging Tools

### Quick Visualization
```python
from debug_utils import DebugUtils

model = MnistModel.load_trained_model()
_, test_loader = model.get_datasets()

# See predictions
DebugUtils.analyze_predictions(model, test_loader, model.device)

# View confusion matrix
DebugUtils.plot_confusion_matrix(model, test_loader, model.device)
```

### Enhanced Training
```python
from debug_model import DebugModel

debug_wrapper = DebugModel()
train_loader, test_loader = debug_wrapper.model.get_datasets()

# Train with detailed logging
debug_wrapper.train_with_debugging(train_loader, test_loader, optimizer, criterion)

# Comprehensive analysis
analysis = debug_wrapper.analyze_model_performance(test_loader)
```

## 📊 Advanced Logging

### TensorBoard
```bash
python train_with_logging.py --tensorboard --epochs 5
tensorboard --logdir=runs
```

### Weights & Biases
```bash
wandb login
python train_with_logging.py --wandb --experiment-name "mnist_v1"
```

### Both Together
```bash
python train_with_logging.py --tensorboard --wandb --epochs 10
```

## 🌐 API Usage

### Start API
```bash
cd api
python api.py
```

### Python Client
```python
import requests

with open("digit.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict", 
        files={"file": f}
    )
    result = response.json()
    print(f"Digit: {result['predicted_digit']}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@digit.png"
```

## 📦 Installation

```bash
# Install dependencies
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

## 🎛️ Command Line Options

### MnistModel.py
```bash
python MnistModel.py --help

# Common options:
--train              # Enable training mode
--epochs 10          # Number of epochs
--learning-rate 0.001 # Learning rate
--batch-size 64      # Batch size
--save-path model.pth # Save location
```

### train_with_logging.py
```bash
python train_with_logging.py --help

# Logging options:
--tensorboard        # Enable TensorBoard
--wandb             # Enable Weights & Biases
--experiment-name   # Experiment name
```

## 🐳 Docker

### Build & Run
```bash
docker build -t mnist-api .
docker run -p 8000:8000 mnist-api
```

### Environment Variables
```bash
# Custom port
docker run -p 9000:8000 -e PORT=8000 mnist-api
```



# API (FastApi)

This folder contains the FastAPI service that exposes the MNIST model inference endpoints.

## Architecture

The API service:
- Loads the trained model from `../model/saved_model.pth`
- Imports prediction functions from `../model/MnistModel.py`
- Exposes REST endpoints for digit prediction
- Handles image upload and preprocessing
- Returns prediction results in JSON format

## Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check and model availability status
- `POST /predict` - Upload image for digit prediction
- `GET /docs` - Interactive API documentation (Swagger UI)

## Local Development

```bash
cd api
pip install -r requirements.txt
python main.py
```

The API will be available at http://localhost:8000

## Docker

```bash
# Build the API service
docker build -t mnist-api .

# Run the API service
docker run -p 8000:8000 mnist-api
```

## Dependencies

- FastAPI for the web framework
- Uvicorn for the ASGI server
- PyTorch for model inference
- Pillow for image processing
- Python-multipart for file uploads

## Manual test:

In one shell, inside "model folder"

```shell
python api.py
```

In another shell, 
```shell
curl -X POST -F 'file=@test_images/mnist_test_07_label_9_idx_8935.png' http://localhost:8000/predict
```

