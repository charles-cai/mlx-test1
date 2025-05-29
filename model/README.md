# MNIST Model Package

Complete MNIST digit recognition system with training, debugging, and API deployment.

## 🚀 Quick Start

### Train a Model
```bash
cd model
python MnistModel.py --train --epochs 5
```

### Run the API (Development)
```bash
cd api
python api.py
# Open http://localhost:8000
```

### Docker (Production)
```bash
./build-dockers.sh --build
# Then
docker compose up
```

## 📁 File Structure
```
model/
├── MnistModel.py      # Main model class (training + prediction)
├── train_with_logging.py  # TensorBoard/W&B training
└── README.md          # This file
api/
├── api.py            # FastAPI web service
└── (model)           # Trained Model and MnsistModel.py etc dependencies will be copied here.
└── README.md         # API documentation
```

## 🎯 Core Usage

### Training
```python
from MnistModel import MnistModel
model = MnistModel()
train_loader, test_loader = model.get_datasets()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
model.train_model(train_loader, optimizer, criterion, num_epochs=5)
model.save_model("my_model.pth")
```

### Prediction
```python
model = MnistModel.load_trained_model("saved_model.pth")
with open("digit_image.png", "rb") as f:
    result = model.predict_digit_from_image(f.read())
    print(f"Predicted: {result['predicted_digit']}")
```

## 🔍 Debugging & Logging

- Use `debug_utils.py` for quick visualization and confusion matrix.
- Use `train_with_logging.py` for TensorBoard or Weights & Biases logging.

## 🌐 API Usage

- Start API: `cd api && python api.py`
- Predict via Python:
```python
import requests
with open("digit.png", "rb") as f:
    r = requests.post("http://localhost:8000/predict", files={"file": f})
    print(r.json())
```
- Predict via cURL:
```bash
curl -X POST "http://localhost:8000/predict" -F "file=@digit.png"
```

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🐳 Docker

```bash
./build-dockers.sh --build
# Then
docker compose up
```

## Notes
- For deployment, model files from /model are copied to /api via build-dockers.sh

