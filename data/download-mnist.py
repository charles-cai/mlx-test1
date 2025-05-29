"""
MNIST Dataset Downloader and Manager

This script downloads the MNIST dataset from Hugging Face,
saves it to the specified directory, and provides functionality
to load and inspect the dataset.

Usage:
    python download-mnist.py

Note: If the dataset directory already exists and is not empty,
      you'll need to remove it before running this script again.
      Use: rm -rf data/.mnist
"""
from datasets import load_dataset, load_from_disk
import os

# Constants
MNIST_DATASET_PATH = "./.mnist"

def download_save_to_disk_mnist_dataset(save_path=MNIST_DATASET_PATH):
    """
    Downloads the MNIST dataset and saves it to the specified location.
    
    Args:
        save_path (str): Path where the dataset will be saved
    
    Returns:
        datasets.DatasetDict: The downloaded dataset
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download the MNIST dataset
    print("Downloading MNIST dataset...")
    dataset = load_dataset("ylecun/mnist")

    # You can inspect the dataset
    print(dataset)

    # Save the dataset to the specified directory
    print(f"Saving dataset to {save_path} folder...")
    dataset.save_to_disk(save_path)

    print("MNIST dataset downloaded and saved successfully.")
    
    # You can verify what was saved
    if os.path.exists(save_path):
        print("\nSaved dataset contents:")
        print(os.listdir(save_path))
    
    return dataset

def load_from_disk_mnist_dataset(dataset_path=MNIST_DATASET_PATH):
    """
    Loads the MNIST dataset from a local directory.
    
    Args:
        dataset_path (str): Path where the dataset is saved
    
    Returns:
        datasets.DatasetDict: The loaded dataset
    """
    # Check if the path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return None
        
    # Load the dataset from your local storage
    print(f"Loading MNIST dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    
    print("\nDataset loaded successfully!")
    
    return dataset


def inspect_mnist_dataset(dataset):
    """
    Inspects and displays information about the MNIST dataset.
    
    Args:
        dataset: The MNIST dataset to inspect
        
    Returns:
        None
    """
    if dataset is None:
        print("No dataset provided to inspect")
        return
        
    # Print structure information
    print(f"Dataset structure: {dataset}")
    
    # Display available splits
    print(f"\nAvailable splits: {list(dataset.keys())}")

    # Show a sample from the training set (if it exists)
    if "train" in dataset:
        print("\nSample from training set:")
        print(dataset["train"][0])
        
        # Display dataset size
        print(f"\nTraining set size: {len(dataset['train'])} examples")
        if "test" in dataset:
            print(f"Test set size: {len(dataset['test'])} examples")

def main():
    """
    Main function to downloading and loading the MNIST dataset.
    """
    # Check if dataset directory exists and is not empty
    if os.path.exists(MNIST_DATASET_PATH) and os.listdir(MNIST_DATASET_PATH):
        print(f"Error: The directory {MNIST_DATASET_PATH} already exists and is not empty.")
        print("Please run the following command to remove it:")
        print(f"\n    rm -rf {MNIST_DATASET_PATH}\n")
        print("Then run this script again.")
        return
    
    # Download and save the dataset
    download_save_to_disk_mnist_dataset()
    
    # Load the dataset from disk
    dataset = load_from_disk_mnist_dataset()
    
    # Inspect the dataset
    if dataset is not None:
        inspect_mnist_dataset(dataset)
        print("\nSuccessfully loaded and inspected the dataset. Ready for use!")

if __name__ == "__main__":
    main()