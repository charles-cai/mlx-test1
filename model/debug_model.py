# filepath: /home/charles/_github/charles-cai/mli-test1/model/debug_model.py
"""
Model Debugging Script

This script loads a trained model and performs various debugging operations
to help understand and troubleshoot model behavior.
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os

from model.trainer import MNISTModel
from model.debug_utils import check_model_outputs, debug_model_gradients

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Debug MNIST model')
    parser.add_argument('--model_path', type=str, default='model/saved_model.pth',
                        help='Path to saved model weights')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--check_gradients', action='store_true',
                        help='Check model gradients')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = MNISTModel().to(device)
    
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
    else:
        print(f"Warning: Model file {args.model_path} not found! Using untrained model.")
    
    # Set up data loaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_ds = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    # Check model outputs
    print("Checking model outputs...")
    check_model_outputs(model, test_loader, device, num_batches=1)
    
    # Check gradients if requested
    if args.check_gradients:
        print("\nChecking model gradients...")
        criterion = nn.CrossEntropyLoss()
        debug_model_gradients(model, criterion, test_loader, device)
    
    print("\nModel debugging complete!")

if __name__ == "__main__":
    main()