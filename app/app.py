import gradio as gr
import numpy as np
import requests
import io
import hashlib
import uuid
from PIL import Image
import os

# Import local PostgresLogger
try:
    from PostgresLogger import PostgresLogger
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"Database import error: {e}")
    DATABASE_AVAILABLE = False

# Global variables
session_id = str(uuid.uuid4())
API_URL = "http://localhost:8889"  # FastAPI endpoint
db_logger = None

# Initialize database logger
if DATABASE_AVAILABLE:
    try:
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/mnist_db")
        db_logger = PostgresLogger(
            database_url=DATABASE_URL,
            model_name="MnistModel",
            session_id=session_id
        )
        print(f"✅ Database logger initialized")
    except Exception as e:
        print(f"❌ Failed to initialize database logger: {e}")
        DATABASE_AVAILABLE = False

def predict_digit(image, true_label=None):
    """
    Predict digit from drawn image using the FastAPI endpoint
    
    Args:
        image: PIL Image or numpy array from Gradio
        true_label: Optional user-provided true label
        
    Returns:
        Tuple of (prediction_text, confidence_chart, database_status)
    """
    # Check if image is provided
    if image is None:
        return "❌ Please draw something first", None, "Image: No image provided"
    
    try:
        # Convert image to PIL format
        pil_image = None
        
        if isinstance(image, np.ndarray):
            print(f"Debug: Received numpy array with shape: {image.shape}, dtype: {image.dtype}")
            
            # Handle different array dimensions
            if len(image.shape) == 3:
                # If it's RGB/RGBA, convert to grayscale
                if image.shape[2] == 3:  # RGB
                    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                elif image.shape[2] == 4:  # RGBA
                    image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
                image = image.astype(np.uint8)
            elif len(image.shape) == 2:
                # Already grayscale
                pass
            else:
                return "❌ Invalid image dimensions", None, "Image: Invalid dimensions"
            
            # Normalize to 0-255 range
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image, mode='L')
            
        elif hasattr(image, 'convert'):
            # It's already a PIL Image
            print(f"Debug: Received PIL Image with mode: {image.mode}, size: {image.size}")
            pil_image = image.convert('L')
            
        else:
            print(f"Debug: Unknown image type: {type(image)}")
            return f"❌ Unsupported image type: {type(image)}", None, "Image: Unsupported type"
        
        # Verify we have a valid PIL image
        if pil_image is None:
            return "❌ Failed to process image", None, "Image: Processing failed"
        
        # Resize to 28x28 for MNIST (if not already)
        if pil_image.size != (28, 28):
            pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
        
        print(f"Debug: Final PIL image - Mode: {pil_image.mode}, Size: {pil_image.size}")
        
        # Convert to bytes for API call
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        print(f"Debug: Image bytes length: {len(image_bytes)}")
        
        # Call FastAPI endpoint
        try:
            files = {'file': ('image.png', image_bytes, 'image/png')}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
            
            if response.status_code != 200:
                return f"❌ API Error: {response.status_code} - {response.text}", None, "API: Error"
            
            result = response.json()
            
        except requests.exceptions.RequestException as e:
            return f"❌ Failed to connect to API: {str(e)}", None, f"API: Connection failed - {API_URL}"
        
        predicted_digit = result['predicted_digit']
        confidence = result['confidence']
        confidence_scores = result['confidence_scores']
        
        # Create image hash for database
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
        # Prepare prediction text
        prediction_text = f"""
🎯 **Predicted Digit: {predicted_digit}**
🎪 **Confidence: {confidence:.1%}**

📊 **All Probabilities:**
"""
        
        # Create confidence chart data
        chart_data = []
        for digit in range(10):
            chart_data.append({
                "digit": str(digit),
                "confidence": float(confidence_scores[str(digit)])
            })
        
        # Database logging
        database_status = "Database: Not configured"
        if DATABASE_AVAILABLE and db_logger:
            try:
                # Log to database using PostgresLogger
                success = db_logger.log_prediction(
                    prediction=float(predicted_digit),
                    true_label=float(true_label) if true_label is not None else None,
                    confidence=confidence,
                    input_shape="(28, 28, 1)",
                    metadata={
                        "confidence_scores": confidence_scores,
                        "image_hash": image_hash,
                        "session_id": session_id
                    }
                )
                
                if success:
                    database_status = "Database: ✅ Logged successfully"
                else:
                    database_status = "Database: ❌ Failed to log"
                    
            except Exception as db_error:
                database_status = f"Database: ❌ Error - {str(db_error)}"
        
        # Add individual probabilities to text
        for digit in range(10):
            conf = float(confidence_scores[str(digit)])
            bar = "█" * int(conf * 20)  # Simple text bar
            prediction_text += f"\n{digit}: {conf:.1%} {bar}"
        
        return prediction_text, chart_data, database_status
        
    except Exception as e:
        print(f"Debug: Exception in predict_digit: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Prediction error: {str(e)}", None, f"Error: {str(e)}"

def update_true_label(image, predicted_text, true_label):
    """Update the database with user-provided true label"""
    if true_label is None:
        return "Please select the correct digit first"
    
    # Re-run prediction with true label
    prediction_text, confidence_chart, database_status = predict_digit(image, int(true_label))
    return f"✅ Updated with true label: {true_label}\n\n{prediction_text}"

def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="MNIST Digit Recognizer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔢 MNIST Digit Recognition")
        gr.Markdown("Draw a digit (0-9) in the canvas below and get AI predictions!")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Drawing canvas
                canvas = gr.Sketchpad(
                    label="✏️ Draw a digit here",
                    type="pil",
                    image_mode="L",
                    canvas_size=(280, 280)
                    # brush_radius=15
                )
                
                # True label input
                true_label = gr.Dropdown(
                    choices=list(range(10)),
                    label="🎯 What digit did you actually draw?",
                    info="Optional: Help improve the model!"
                )
                
                # Buttons
                with gr.Row():
                    predict_btn = gr.Button("🔮 Predict Digit", variant="primary")
                    update_btn = gr.Button("📝 Update True Label", variant="secondary")
                    clear_btn = gr.Button("🗑️ Clear Canvas")
            
            with gr.Column(scale=1):
                # Results display
                prediction_output = gr.Markdown(
                    label="🎯 Prediction Results",
                    value="Draw a digit and click 'Predict Digit' to see results!"
                )
                
                # Confidence chart
                confidence_chart = gr.BarPlot(
                    x="digit",
                    y="confidence",
                    title="📊 Confidence Scores by Digit",
                    x_title="Digit",
                    y_title="Confidence",
                    height=300
                )
                
                # Database status
                db_status = gr.Textbox(
                    label="💾 Database Status",
                    value="Ready to log predictions...",
                    interactive=False
                )
        
        # Event handlers
        predict_btn.click(
            fn=lambda img, true_lbl: predict_digit(img, true_lbl),
            inputs=[canvas, true_label],
            outputs=[prediction_output, confidence_chart, db_status]
        )
        
        update_btn.click(
            fn=lambda img, pred_text, true_lbl: update_true_label(img, pred_text, true_lbl),
            inputs=[canvas, prediction_output, true_label],
            outputs=[prediction_output]
        )
        
        clear_btn.click(
            fn=lambda: (None, "Draw a digit and click 'Predict Digit' to see results!", None, "Ready to log predictions..."),
            outputs=[canvas, prediction_output, confidence_chart, db_status]
        )
        
        # Add instructions
        gr.Markdown("""
        ## 📝 Instructions:
        1. **Draw** a digit (0-9) in the canvas using your mouse
        2. **Click** "Predict Digit" to get AI prediction
        3. **Optionally** select the correct digit and click "Update True Label" to help improve the model
        4. **Clear** the canvas to start over
        
        ## 🔍 Features:
        - 🎯 Real-time digit prediction with confidence scores
        - 📊 Visual confidence chart for all digits
        - 💾 Automatic logging to PostgreSQL database
        - 📝 User feedback collection for model improvement
        """)
    
    return interface

def main():
    """Main function to launch the Gradio app"""
    print(f"🚀 Starting MNIST Gradio Interface...")
    print(f"📡 API Endpoint: {API_URL}")
    print(f"💾 Database available: {DATABASE_AVAILABLE}")
    print(f"🔑 Session ID: {session_id}")
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Standard Gradio port
        share=False,            # Set to True for public sharing
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()