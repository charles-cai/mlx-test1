import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from PIL import Image
import io
from typing import Tuple, Dict, Optional
import hashlib
import uuid
from datetime import datetime
import json

from datasets import load_from_disk
from torchvision import datasets as tv_datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


class MnistModel(nn.Module):
    """
    MNIST CNN Model with integrated training, evaluation, and prediction capabilities
    """
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(9216, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.net(x)

    def get_datasets(self, from_huggingface=True, data_path='../data', batch_size_train=64, 
                     batch_size_test=1000, hf_path=None) -> Tuple[DataLoader, DataLoader]:
        """
        Load MNIST datasets either from Hugging Face or torchvision
        
        Args:
            from_huggingface (bool): Whether to load from Hugging Face datasets
            data_path (str): Path for data storage/loading
            batch_size_train (int): Batch size for training data
            batch_size_test (int): Batch size for testing data
            hf_path (str): Specific path for Hugging Face dataset, if None uses data_path/.mnist
            
        Returns:
            tuple: (train_loader, test_loader) - DataLoaders for training and testing
        """
        if from_huggingface:
            if hf_path is None:
                hf_path = f"{data_path}/.mnist"
            
            try:
                mnist_dataset = load_from_disk(hf_path)
                
                # Convert Hugging Face datasets to PyTorch datasets
                def process_hf_dataset(dataset):
                    images = np.array(dataset['image']).reshape(-1, 1, 28, 28) / 255.0
                    images = (images - 0.1307) / 0.3081  # Apply the same normalization
                    labels = np.array(dataset['label'])
                    return TensorDataset(torch.tensor(images, dtype=torch.float32), 
                                        torch.tensor(labels, dtype=torch.long))
                
                train_ds = process_hf_dataset(mnist_dataset['train'])
                test_ds = process_hf_dataset(mnist_dataset['test'])
                
            except Exception as e:
                print(f"Error loading from Hugging Face: {e}")
                print("Falling back to torchvision datasets")
                from_huggingface = False
        
        if not from_huggingface:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_ds = tv_datasets.MNIST(data_path, train=True, download=True, transform=transform)
            test_ds = tv_datasets.MNIST(data_path, train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size_test)
        
        return train_loader, test_loader

    def train_model(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module, num_epochs: int = 5) -> 'MnistModel':
        """Train the model"""
        for epoch in range(1, num_epochs + 1):
            self.train()
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self(images), labels)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch} done")
        return self

    def train_model_with_logging(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                                criterion: nn.Module, num_epochs: int = 5, 
                                use_tensorboard: bool = False, use_wandb: bool = False,
                                experiment_name: str = "mnist_training") -> 'MnistModel':
        """
        Train the model with optional TensorBoard and/or Weights & Biases logging
        
        Args:
            train_loader: DataLoader for training data
            optimizer: PyTorch optimizer
            criterion: Loss function
            num_epochs: Number of training epochs
            use_tensorboard: Whether to log to TensorBoard
            use_wandb: Whether to log to Weights & Biases
            experiment_name: Name for the experiment
        """
        # Initialize logging
        writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(f'runs/{experiment_name}')
                print("TensorBoard logging enabled")
            except ImportError:
                print("TensorBoard not available. Install with: pip install tensorboard")
                use_tensorboard = False
        
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project="mnist-training",
                    name=experiment_name,
                    config={
                        "epochs": num_epochs,
                        "optimizer": type(optimizer).__name__,
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "batch_size": train_loader.batch_size,
                        "criterion": type(criterion).__name__
                    }
                )
                # Log model architecture
                wandb.watch(self, log="all")
                print("Weights & Biases logging enabled")
            except ImportError:
                print("Weights & Biases not available. Install with: pip install wandb")
                use_wandb = False
        
        global_step = 0
        
        for epoch in range(1, num_epochs + 1):
            self.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                epoch_loss += loss.item()
                global_step += 1
                
                # Log every 100 batches
                if batch_idx % 100 == 0:
                    batch_acc = 100. * correct / total
                    
                    if use_tensorboard and writer:
                        writer.add_scalar('Training/Loss', loss.item(), global_step)
                        writer.add_scalar('Training/Accuracy', batch_acc, global_step)
                        
                        # Log gradients
                        for name, param in self.named_parameters():
                            if param.grad is not None:
                                writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                                writer.add_histogram(f'Parameters/{name}', param, global_step)
                    
                    if use_wandb:
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/accuracy": batch_acc,
                            "epoch": epoch,
                            "step": global_step
                        })
                    
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}, Acc: {batch_acc:.2f}%')
            
            # Epoch-level logging
            avg_loss = epoch_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            if use_tensorboard and writer:
                writer.add_scalar('Epoch/Loss', avg_loss, epoch)
                writer.add_scalar('Epoch/Accuracy', epoch_acc, epoch)
            
            if use_wandb:
                wandb.log({
                    "epoch/loss": avg_loss,
                    "epoch/accuracy": epoch_acc,
                    "epoch": epoch
                })
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}, Accuracy = {epoch_acc:.2f}%")
        
        # Cleanup
        if use_tensorboard and writer:
            writer.close()
            print(f"TensorBoard logs saved in runs/{experiment_name}")
            print("View with: tensorboard --logdir=runs")
        
        if use_wandb:
            wandb.finish()
        
        return self

    def evaluate_model(self, test_loader: DataLoader) -> float:
        """Evaluate the model and return accuracy"""
        self.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                preds = self(images.to(self.device)).argmax(dim=1)
                correct += (preds == labels.to(self.device)).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        print(f"Test accuracy: {accuracy:.4f}")
        return accuracy

    def save_model(self, path: str):
        """Save the model to disk"""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model from disk"""
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.eval()
        print(f"Model loaded from {path}")

    def preprocess_image_for_prediction(self, image_bytes: bytes) -> torch.Tensor:
        """
        Preprocess uploaded image to match MNIST format:
        - Convert to grayscale
        - Resize to 28x28
        - Normalize using MNIST mean and std
        - Convert to tensor
        """
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28))
        
        # Convert to numpy array and normalize to [0, 1]
        image_array = np.array(image) / 255.0
        
        # Apply MNIST normalization (mean=0.1307, std=0.3081)
        image_array = (image_array - 0.1307) / 0.3081
        
        # Convert to tensor and add batch and channel dimensions
        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)
        
        return image_tensor

    def predict_digit_from_image(self, image_bytes: bytes) -> Dict:
        """
        Complete prediction pipeline:
        - Preprocess image
        - Make prediction
        - Return results with confidence scores
        """
        # Preprocess image
        image_tensor = self.preprocess_image_for_prediction(image_bytes)
        image_tensor = image_tensor.to(self.device)
        
        # Make prediction
        self.eval()
        with torch.no_grad():
            outputs = self(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence_scores = probabilities.cpu().numpy()[0]
            predicted_digit = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])
            confidence = float(confidence_scores[predicted_digit])
        
        # Create image hash for reference
        image_hash = hashlib.md5(image_bytes).hexdigest()
        
        # Prepare response
        result = {
            "predicted_digit": predicted_digit,
            "confidence": confidence,
            "confidence_scores": {
                str(i): float(score) for i, score in enumerate(confidence_scores)
            },
            "image_hash": image_hash
        }
        
        return result

    @classmethod
    def load_trained_model(cls, model_path: str = "saved_model.pth") -> 'MnistModel':
        """Load a trained MNIST model"""
        model = cls()
        model.load_model(model_path)
        return model

    @staticmethod
    def display_config(args):
        """Display all configuration values being used"""
        print("=" * 50)
        print("CONFIGURATION")
        print("=" * 50)
        # Use argparse's built-in functionality to display all arguments
        for arg, value in vars(args).items():
            print(f"{arg.replace('_', '-')}: {value}")
        print("=" * 50)


# Standalone functions for API compatibility
def predict_digit_from_image(image_bytes: bytes, model_path: str = "saved_model.pth") -> Dict:
    """
    Standalone function for API compatibility
    Complete prediction pipeline
    """
    model = MnistModel.load_trained_model(model_path)
    return model.predict_digit_from_image(image_bytes)


def main(args):
    """Main function to run the training pipeline or evaluation"""
    # Display configuration
    MnistModel.display_config(args)
    
    # Initialize model
    model = MnistModel()
    print(f"Using device: {model.device}")
    
    # Get data
    train_loader, test_loader = model.get_datasets(
        from_huggingface=args.huggingface,
        data_path=args.data_path,
        batch_size_train=args.batch_size,
        batch_size_test=args.test_batch_size
    )
    
    if args.train:
        print("Training mode")
        # Initialize optimization tools
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        model.train_model(train_loader, optimizer, criterion, args.epochs)
        
        # Evaluate
        model.evaluate_model(test_loader)
        
        # Save
        model.save_model(args.save_path)
    else:
        print("Evaluation mode")
        # Load existing model
        try:
            model.load_model(args.save_path)
        except FileNotFoundError:
            print(f"Error: Model file {args.save_path} not found. Please train the model first.")
            return
        
        # Evaluate
        model.evaluate_model(test_loader)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MNIST Training Script')
    parser.add_argument('--train', action='store_true', default=False, help='Train the model (default: False, evaluation mode)')
    parser.add_argument('--huggingface', action='store_true', default=True, help='Use HuggingFace dataset (default: True)')
    parser.add_argument('--data-path', type=str, default='../data', help='Path to the dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='Test batch size (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--save-path', type=str, default='saved_model.pth', help='Path to save the model')
    parser.add_argument('--cpu', action='store_true', default=False, help='Force CPU training')
    
    args = parser.parse_args()
    main(args)