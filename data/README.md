## Key Dependencies

- `datasets`: For downloading and handling the Hugging Face datasets
- `Pillow`: Required for working with image data

## Usage

### Downloading the MNIST Dataset

Run the downloader script to fetch the MNIST dataset from Hugging Face and save it locally:

```bash
python data/download-mnist.py
```

This will:
1. Download the MNIST dataset
2. Save it to `data/.mnist` directory
3. Load it back and display dataset information

### Script Features

The script (`download-mnist.py`) provides several functions:

1. `download_save_to_disk_mnist_dataset()`: Downloads the MNIST dataset and saves it locally
2. `load_from_disk_mnist_dataset()`: Loads the dataset from the local directory
3. `inspect_mnist_dataset()`: Displays information about the dataset structure and contents

### Using the Dataset in Your Code

After downloading, you can load the dataset in your own code:

```python
from datasets import load_from_disk

# Load the dataset from the local directory
dataset = load_from_disk("data/.mnist")

# Access training data
train_data = dataset["train"]

# Access test data
test_data = dataset["test"]

# Get a sample
sample = train_data[0]
```

## Troubleshooting

### Directory Already Exists

If you see an error like:
```
Error: The directory data/.mnist already exists and is not empty.
```

Run the following command to remove the existing dataset, then try again:
```bash
rm -rf data/.mnist
```

### Missing Pillow

If you encounter:
```
ImportError: To support decoding images, please install 'Pillow'.
```

Make sure to install the Pillow package:
```bash
pip install Pillow
```

## Dataset Structure

The MNIST dataset contains:
- Training set: 60,000 examples
- Test set: 10,000 examples

Each example contains:
- `image`: 28x28 pixel grayscale image of a handwritten digit
- `label`: The corresponding digit (0-9)

## License

his utility code is provided under MIT license. The MNIST dataset itself has its own license terms.