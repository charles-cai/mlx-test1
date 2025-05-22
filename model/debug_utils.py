# filepath: /home/charles/_github/charles-cai/mli-test1/model/debug_utils.py
"""
Debugging utilities for the MNIST model trainer.
This file provides functions to help with debugging the model during training.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


def visualize_batch(batch, predictions=None):
    """
    Visualize a batch of images with optional predictions.
    
    Args:
        batch: Tuple of (images, labels)
        predictions: Optional tensor of predicted labels
    """
    images, labels = batch
    
    # Convert images from tensor to numpy for display
    if isinstance(images, torch.Tensor):
        # Convert to numpy and reshape
        images = images.cpu().numpy()
    
    # Create a grid of images
    batch_size = min(25, images.shape[0])  # Show up to 25 images
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    
    plt.figure(figsize=(10, 10))
    for i in range(batch_size):
        plt.subplot(grid_size, grid_size, i + 1)
        # Reshape and denormalize if necessary
        img = images[i].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        
        # Add label or prediction
        title = f"True: {labels[i]}"
        if predictions is not None:
            pred = predictions[i]
            title += f" | Pred: {pred}"
            # Highlight incorrect predictions
            if pred != labels[i]:
                plt.title(title, color='red')
            else:
                plt.title(title, color='green')
        else:
            plt.title(title)
            
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def check_model_outputs(model, dataloader, device, num_batches=1):
    """
    Examine model outputs on a few batches to check for potential issues.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for the dataset
        device: Device to run the model on
        num_batches: Number of batches to check
    """
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Check for NaN values
            if torch.isnan(outputs).any():
                print("Warning: NaN values in model outputs!")
                
            # Print output statistics
            print(f"Batch {i+1} statistics:")
            print(f"  Output min: {outputs.min().item():.4f}")
            print(f"  Output max: {outputs.max().item():.4f}")
            print(f"  Output mean: {outputs.mean().item():.4f}")
            print(f"  Output std: {outputs.std().item():.4f}")
            
            # Get predictions
            predictions = outputs.argmax(dim=1).cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            # Show confusion statistics
            correct = (predictions == labels_cpu).sum()
            print(f"  Correct predictions: {correct}/{len(predictions)} ({correct/len(predictions)*100:.2f}%)")
            
            # Visualize the batch with predictions
            visualize_batch((images.cpu(), labels_cpu), predictions)


def debug_model_gradients(model, criterion, dataloader, device):
    """
    Debug model gradients to check for issues like vanishing/exploding gradients.
    
    Args:
        model: PyTorch model
        criterion: Loss function
        dataloader: DataLoader for the dataset
        device: Device to run the model on
    """
    model.train()
    
    # Get a batch of data
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("Gradient statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            print(f"  {name}:")
            print(f"    Min: {grad.min().item():.8f}")
            print(f"    Max: {grad.max().item():.8f}")
            print(f"    Mean: {grad.mean().item():.8f}")
            print(f"    Std: {grad.std().item():.8f}")
            
            # Check for very small or large gradients
            if grad.abs().max() > 10:
                print(f"    Warning: Large gradients detected in {name}!")
            if grad.abs().max() < 1e-6:
                print(f"    Warning: Very small gradients detected in {name}!")


if __name__ == "__main__":
    # Example usage - for testing the debug utilities
    from torchvision import datasets, transforms
    from model.trainer import MNISTModel
    
    # Setup data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MNISTModel().to(device)
    
    try:
        # Try to load saved model if it exists
        model.load_state_dict(torch.load('model/saved_model.pth'))
        print("Loaded saved model")
    except FileNotFoundError:
        print("No saved model found, using untrained model")
    
    # Check model outputs
    check_model_outputs(model, test_loader, device)