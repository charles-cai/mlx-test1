from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import sys
import os
import base64
import numpy as np
import io
from pathlib import Path
from typing import Dict
from PIL import Image

# Add the parent directory to sys.path to import the model
sys.path.append(str(Path(__file__).parent.parent))

from model.MnistModel import MnistModel

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Digit Prediction API",
    description="API for predicting handwritten digits using a trained CNN model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

# Pydantic models for request/response
class NumpyImageRequest(BaseModel):
    image_data: str  # base64 encoded numpy array
    shape: list      # [height, width] or [height, width, channels]
    dtype: str       # numpy dtype as string (e.g., 'uint8', 'float32')

class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    confidence_scores: Dict[str, float]
    image_hash: str

@app.on_event("startup")
async def startup_event():
    """Load the trained model on startup"""
    global model
    try:
        model_path = os.getenv("MODEL_PATH", "../model/saved_model.pth")
        model = MnistModel.load_trained_model(model_path)
        print(f"✅ Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        # Don't raise here, let the health check handle it

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MNIST Digit Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict (upload image file)",
            "predict_numpy": "/predict-numpy (send numpy array as base64)",
            "health": "/health",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(model.device) if model else "unknown"
    }

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)) -> Dict:
    """
    Predict digit from uploaded image
    
    Args:
        file: Image file (PNG, JPG, JPEG)
        
    Returns:
        Prediction results with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Make prediction
        result = model.predict_digit_from_image(image_bytes)
        
        # Add additional metadata
        result["filename"] = file.filename
        result["file_size"] = len(image_bytes)
        result["content_type"] = file.content_type
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-numpy")
async def predict_digit_numpy(request: NumpyImageRequest) -> Dict:
    """
    Predict digit from numpy array sent as base64
    
    Args:
        request: NumpyImageRequest containing base64 encoded numpy array
        
    Returns:
        Prediction results with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode base64 numpy array
        image_bytes = base64.b64decode(request.image_data)
        image_array = np.frombuffer(image_bytes, dtype=getattr(np, request.dtype))
        
        # Reshape array
        if len(request.shape) == 2:
            image_array = image_array.reshape(request.shape)
        elif len(request.shape) == 3:
            image_array = image_array.reshape(request.shape)
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid shape: {request.shape}. Must be 2D or 3D."
            )
        
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 3:  # RGB
                image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            elif image_array.shape[2] == 4:  # RGBA
                image_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            image_array = image_array.astype(np.uint8)
        
        # Ensure uint8 range
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        
        # Convert to PIL and then to bytes (to reuse existing model method)
        pil_image = Image.fromarray(image_array, mode='L')
        
        # Resize to 28x28 if needed
        if pil_image.size != (28, 28):
            pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        image_bytes_processed = img_byte_arr.getvalue()
        
        # Make prediction using existing model method
        result = model.predict_digit_from_image(image_bytes_processed)
        
        # Add metadata
        result["input_shape"] = request.shape
        result["input_dtype"] = request.dtype
        result["processed_size"] = pil_image.size
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Numpy prediction failed: {str(e)}"
        )

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "CNN",
        "input_shape": [1, 28, 28],
        "output_classes": 10,
        "device": str(model.device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MNIST Prediction API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8889, help='Port to bind to')
    parser.add_argument('--model-path', type=str, default='../model/saved_model.pth', help='Path to the trained model')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    # Set model path environment variable
    os.environ["MODEL_PATH"] = args.model_path
    
    print(f"Starting MNIST Prediction API on {args.host}:{args.port}")
    print(f"Model path: {args.model_path}")
    print(f"Documentation available at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api:app" if not args.reload else "api.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )