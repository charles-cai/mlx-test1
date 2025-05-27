from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import prediction function from the same directory
from MnistModel import predict_digit_from_image

app = FastAPI(
    title="MNIST Digit Recognition API",
    description="API for predicting handwritten digits using a CNN model",
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

@app.on_event("startup")
async def startup_event():
    """Check if model functionality is available"""
    print("MNIST API started successfully")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "MNIST Digit Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for digit prediction", 
            "/health": "GET - Check API health",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_path = Path(__file__).parent / "saved_model.pth"
    return {
        "status": "healthy" if model_path.exists() else "unhealthy",
        "model_available": model_path.exists()
    }

@app.post("/predict")
async def predict_uploaded_image(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image
    
    Args:
        file: Image file (PNG, JPG, etc.)
        
    Returns:
        JSON with predicted digit, confidence, and all confidence scores
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Use MnistModel for prediction (same directory)
        model_path = Path(__file__).parent / "saved_model.pth"
        result = predict_digit_from_image(image_bytes, str(model_path))
        
        return JSONResponse(content=result)
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model file not found. Please train the model first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)