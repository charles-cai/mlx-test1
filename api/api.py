from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
import os
from pathlib import Path
from typing import Dict

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
            "predict": "/predict",
            "health": "/health",
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