import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from datasets import load_from_disk
from torchvision import datasets as tv_datasets, transforms
from torch.utils.data import DataLoader, TensorDataset


# 1) Define simple CNN
class MNISTModel(nn.Module):
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

    def forward(self, x):
        return self.net(x)


# 2) Data loaders
def get_datasets(from_huggingface=False, data_path='../data', batch_size_train=64, 
                 batch_size_test=1000, hf_path=None):
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


# 3) Training function
def train_model(model, train_loader, optimizer, criterion, device, num_epochs=5):
    """Train the model"""
    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} done")
    return model


# 4) Evaluation function
def evaluate_model(model, test_loader, device):
    """Evaluate the model and return accuracy"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(device)).argmax(dim=1)
            correct += (preds == labels.to(device)).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}")
    return accuracy


# 5) Save model function
def save_model(model, path):
    """Save the model to disk"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# 6) Display configuration function
def display_config(args):
    """Display all configuration values being used"""
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    # Use argparse's built-in functionality to display all arguments
    for arg, value in vars(args).items():
        print(f"{arg.replace('_', '-')}: {value}")
    print("=" * 50)


def main(args):
    """Main function to run the training pipeline or evaluation"""
    # Display configuration
    display_config(args)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Get data
    train_loader, test_loader = get_datasets(
        from_huggingface=args.huggingface,
        data_path=args.data_path,
        batch_size_train=args.batch_size,
        batch_size_test=args.test_batch_size
    )
    
    # Initialize model
    model = MNISTModel().to(device)
    
    if args.train:
        print("Training mode")
        # Initialize optimization tools
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Train
        model = train_model(model, train_loader, optimizer, criterion, device, args.epochs)
        
        # Evaluate
        evaluate_model(model, test_loader, device)
        
        # Save
        save_model(model, args.save_path)
    else:
        print("Evaluation mode")
        # Load existing model
        try:
            model.load_state_dict(torch.load(args.save_path, map_location=device))
            print(f"Model loaded from {args.save_path}")
        except FileNotFoundError:
            print(f"Error: Model file {args.save_path} not found. Please train the model first.")
            return
        
        # Evaluate
        evaluate_model(model, test_loader, device)


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
