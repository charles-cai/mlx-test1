"""
Random MNIST test image selector and console display
Selects 10 random images from MNIST test dataset and displays them in console
"""
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import random
import os
import sys
from pathlib import Path
from PIL import Image

def console_display_image(image_array, label=None, width=28):
    """
    Display a 28x28 image in console using ASCII characters
    
    Args:
        image_array: numpy array of shape (28, 28) with values 0-255
        label: optional label to display
        width: width of the image (should be 28 for MNIST)
    """
    # Normalize to 0-255 if needed
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    
    # ASCII characters from dark to light
    ascii_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    
    print(f"\n{'='*width*2}")
    if label is not None:
        print(f"Label: {label}")
    print(f"{'='*width*2}")
    
    for row in image_array:
        line = ""
        for pixel in row:
            # Map pixel value (0-255) to ASCII character index
            char_idx = int((pixel / 255.0) * (len(ascii_chars) - 1))
            char_idx = min(char_idx, len(ascii_chars) - 1)
            line += ascii_chars[char_idx] * 2  # Double width for better aspect ratio
        print(line)
    print(f"{'='*width*2}\n")

def load_mnist_test_data(data_path="../data"):
    """
    Load MNIST test dataset
    
    Args:
        data_path: Path to data folder
        
    Returns:
        test_dataset: MNIST test dataset
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs(data_path, exist_ok=True)
        
        # Transform to convert to tensor but keep in 0-255 range for display
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Load test dataset
        test_dataset = datasets.MNIST(
            root=data_path, 
            train=False, 
            download=True, 
            transform=transform
        )
        
        print(f"✅ Loaded MNIST test dataset: {len(test_dataset)} images")
        return test_dataset
        
    except Exception as e:
        print(f"❌ Error loading MNIST data: {e}")
        return None

def select_random_images(dataset, num_images=10):
    """
    Select random images from dataset
    
    Args:
        dataset: MNIST dataset
        num_images: Number of images to select
        
    Returns:
        List of (image, label, index) tuples
    """
    if dataset is None or len(dataset) == 0:
        return []
    
    # Get random indices
    total_images = len(dataset)
    random_indices = random.sample(range(total_images), min(num_images, total_images))
    
    selected_images = []
    for idx in random_indices:
        image, label = dataset[idx]
        # Convert tensor to numpy array (28, 28)
        image_np = image.squeeze().numpy() * 255  # Convert back to 0-255 range
        selected_images.append((image_np.astype(np.uint8), label, idx))
    
    return selected_images

def save_test_images(selected_images, output_dir="test_images"):
    """
    Save selected images as PNG files for API testing
    
    Args:
        selected_images: List of (image, label, index) tuples
        output_dir: Directory to save images
        
    Returns:
        List of saved image paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []
    
    for i, (image_array, label, idx) in enumerate(selected_images):
        # Create filename
        filename = f"mnist_test_{i:02d}_label_{label}_idx_{idx}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save as PNG
        image_pil = Image.fromarray(image_array, mode='L')
        image_pil.save(filepath)
        saved_paths.append(filepath)
        
        print(f"💾 Saved: {filename}")
    
    return saved_paths

def main():
    """Main function to run the random image selector"""
    print("🎲 MNIST Random Test Image Selector")
    print("="*50)
    
    # Set random seed for reproducibility (optional)
    random.seed(42)
    torch.manual_seed(42)
    
    # Load MNIST test data
    data_path = "../data"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    
    dataset = load_mnist_test_data(data_path)
    if dataset is None:
        print("❌ Failed to load dataset")
        return 1
    
    # Select random images
    num_images = 10
    if len(sys.argv) > 2:
        try:
            num_images = int(sys.argv[2])
        except ValueError:
            print("⚠️  Invalid number of images, using default: 10")
            num_images = 10
    
    print(f"🔍 Selecting {num_images} random test images...")
    selected_images = select_random_images(dataset, num_images)
    
    if not selected_images:
        print("❌ No images selected")
        return 1
    
    # Display images in console
    print(f"\n📺 Displaying {len(selected_images)} random MNIST test images:")
    for i, (image_array, label, idx) in enumerate(selected_images):
        print(f"\n🖼️  Image {i+1}/{len(selected_images)} (Dataset index: {idx})")
        console_display_image(image_array, label)
    
    # Save images for API testing
    print("💾 Saving images for API testing...")
    saved_paths = save_test_images(selected_images)
    
    print(f"\n✅ Successfully processed {len(selected_images)} images")
    print(f"📁 Images saved to: test_images/")
    print(f"🧪 Use these images to test the API:")
    
    for path in saved_paths[:3]:  # Show first 3 examples
        print(f"   curl -X POST -F 'file=@{path}' http://localhost:8000/predict")
    
    if len(saved_paths) > 3:
        print(f"   ... and {len(saved_paths) - 3} more images")
    
    return 0

if __name__ == "__main__":
    exit(main())