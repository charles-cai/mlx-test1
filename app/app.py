import gradio as gr
import numpy as np
import requests
import io
import os
import uuid
import base64
from PIL import Image
try:
    from PostgresLogger import PostgresLogger
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"Database import error: {e}")
    DATABASE_AVAILABLE = False

API_URL = "http://localhost:8889"
db_logger = None
current_prediction = None
current_confidence = None
if DATABASE_AVAILABLE:
    try:
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/mnist_db")
        db_logger = PostgresLogger(
            database_url=DATABASE_URL,
            model_name="MnistModel"
        )
        print(f"✅ Database logger initialized with URL: {DATABASE_URL}")
    except Exception as e:
        print(f"❌ Failed to initialize database logger: {e}")
        DATABASE_AVAILABLE = False

def predict_with_numpy_api(image_array: np.ndarray) -> dict:
    try:
        image_bytes = image_array.tobytes()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        request_data = {
            "image_data": image_b64,
            "shape": list(image_array.shape),
            "dtype": str(image_array.dtype)
        }
        
        response = requests.post(f"{API_URL}/predict-numpy", json=request_data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code}")
            
    except Exception as e:
        raise Exception(f"Numpy API call failed: {str(e)}")

def predict_digit(image):
    global current_prediction, current_confidence

    try:
        pil_image = image["composite"]

        if pil_image is None:
            return "❌ Please draw something first", "0%", get_history_table(), ""

#        pil_image = image.convert('L')

        if pil_image.getextrema() == (0, 0):
            return "❌ Please draw something", "0%", get_history_table(), ""
        
        if pil_image.size != (28, 28):
            pil_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
        
        print(f"Debug: Final PIL image - Mode: {pil_image.mode}, Size: {pil_image.size}")
        
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        print(f"Debug: Image bytes length: {len(image_bytes)}")
        
        try:
            files = {'file': ('image.png', image_bytes, 'image/png')}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get('predicted_digit', 'Unknown')
                confidence = result.get('confidence', 0.0) * 100
                
                current_prediction = prediction
                current_confidence = confidence
                
                prediction_text = f"{prediction}"
                confidence_text = f"{confidence:.1f}%"
                
                return prediction_text, confidence_text, get_history_table(), ""
            else:
                return f"❌ API Error: {response.status_code}", "0%", get_history_table(), ""
                
        except requests.exceptions.RequestException as e:
            return f"❌ Connection Error: {str(e)}", "0%", get_history_table(), ""
        
    except Exception as e:
        print(f"Error in predict_digit: {e}")
        return f"❌ Error: {str(e)}", "0%", get_history_table(), ""

def submit_prediction(true_label_input):
    global current_prediction, current_confidence
    
    if current_prediction is None:
        return get_history_table()
    
    true_label = None
    if true_label_input and str(true_label_input).strip():
        try:
            true_label = int(true_label_input)
            if not (0 <= true_label <= 9):
                print(f"Invalid true label: {true_label}. Must be 0-9")
                return get_history_table()
        except ValueError:
            print(f"Invalid true label format: {true_label_input}")
            return get_history_table()
    
    # Log to database
    if DATABASE_AVAILABLE and db_logger:
        try:
            success = db_logger.log_prediction(
                prediction=current_prediction,
                confidence=current_confidence,
                true_label=true_label
            )
            if success:
                print(f"✅ Logged: prediction={current_prediction}, confidence={current_confidence}, label={true_label}")
            else:
                print("❌ Failed to log prediction")
        except Exception as e:
            print(f"❌ Database error: {e}")
    
    return get_history_table()

def get_history_table():
    """
    Get recent predictions from database as formatted table
    
    Returns:
        List of lists for Gradio DataFrame
    """
    if not DATABASE_AVAILABLE or not db_logger:
        return [["Database not available", "", "", ""]]
    
    try:
        records = db_logger.get_recent_predictions(limit=100)
        if not records:
            return [["No predictions yet", "", "", ""]]
        
        # Format for Gradio table: [timestamp, prediction, confidence, label]
        table_data = []
        for record in records:
            timestamp = record['created_at'][:19].replace('T', ' ')  # Remove microseconds and timezone
            pred = str(record['predicted_digit'])
            conf = f"{record['confidence']:.1f}%"
            label = str(record['label']) if record['label'] is not None else ""
            table_data.append([timestamp, pred, conf, label])
        
        return table_data
        
    except Exception as e:
        print(f"Error getting history: {e}")
        return [["Error loading history", "", "", ""]]
    
def clear_canvas():
    black_image = Image.new('L', (280, 280), 0) 
    return {
        "background": black_image, 
        "layers": None,
        "composite": black_image
    }

def create_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="MNIST Digit Recognizer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔢 MNIST Digit Recognizer")
        gr.Markdown("Draw a digit (0-9) and get AI predictions!")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Drawing canvas with black brush
                canvas = gr.ImageEditor(
                    value = clear_canvas(),
                    label="✏️ Draw a digit here",
                    type="pil",
                    image_mode="L",
                    brush = gr.Brush(default_color="white", default_size=10),
                    canvas_size=(280, 280),
                    eraser=gr.Eraser(default_size=10),
                )

                # Predict button
                predict_btn = gr.Button("🔮 Predict Digit", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                # Results section
                gr.Markdown("### Results")
                
                prediction_display = gr.Textbox(
                    label="Prediction",
                    value="",
                    interactive=False,
                    container=True
                )
                confidence_display = gr.Textbox(
                    label="Confidence",
                    value="",
                    interactive=False,
                    container=True
                )
                true_label = gr.Textbox(
                    label="True Label (0-9)",
                    placeholder="Enter correct digit (optional)",
                    value="",
                    max_lines=1
                )
                
                # Submit button
                submit_btn = gr.Button("📝 Submit", variant="secondary")
        
        # History section
        gr.Markdown("### History")
        history_table = gr.Dataframe(
            headers=["Timestamp", "Prediction", "Confidence", "Label"],
            datatype=["str", "str", "str", "str"],
            value=get_history_table(),
            interactive=False,
            row_count=100,
            col_count=(4, "fixed")
        )
        
        # fix clear button will restore the ImageEditor to be white background
        # canvas.clear(clear_canvas)

        # Event handlers
        predict_btn.click(
            fn=predict_digit,
            inputs=[canvas],
            outputs=[prediction_display, confidence_display, history_table, true_label]
        )
        
        submit_btn.click(
            fn=submit_prediction,
            inputs=[true_label],
            outputs=[history_table]
        )
    
    return interface

def main():
    """Main function to launch the Gradio app"""
    print(f"🚀 Starting MNIST Gradio Interface...")
    print(f"📡 API Endpoint: {API_URL}")
    print(f"💾 Database available: {DATABASE_AVAILABLE}")
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()