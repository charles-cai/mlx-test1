#!/usr/bin/env python3
"""
Advanced training script with TensorBoard and Weights & Biases logging
"""

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from MnistModel import MnistModel


def main():
    parser = argparse.ArgumentParser(description='MNIST Training with Logging')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--experiment-name', type=str, default='mnist_training', help='Experiment name')
    parser.add_argument('--save-path', type=str, default='saved_model.pth', help='Model save path')
    
    args = parser.parse_args()
    
    # Create model
    model = MnistModel()
    print(f"Using device: {model.device}")
    
    # Get data
    train_loader, test_loader = model.get_datasets(batch_size_train=args.batch_size)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Starting training for {args.epochs} epochs...")
    print(f"TensorBoard: {'Enabled' if args.tensorboard else 'Disabled'}")
    print(f"Weights & Biases: {'Enabled' if args.wandb else 'Disabled'}")
    
    # Train with logging
    model.train_model_with_logging(
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=args.epochs,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        experiment_name=args.experiment_name
    )
    
    # Final evaluation
    print("\nFinal evaluation:")
    accuracy = model.evaluate_model(test_loader)
    
    # Save model
    model.save_model(args.save_path)
    
    print(f"\nTraining completed! Final accuracy: {accuracy:.4f}")
    
    if args.tensorboard:
        print(f"\nView TensorBoard logs with:")
        print(f"tensorboard --logdir=runs/{args.experiment_name}")
        print(f"Then open http://localhost:6006")


if __name__ == "__main__":
    main()