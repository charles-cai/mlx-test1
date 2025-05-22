## Set up

My environment is WSL Ubuntu 22.04 LTS on Windows 11

1. Create GitHub repo `mli-test1` under my account https://github.com/charles-cai

```shell
mkdir -p ~/_github/charles-cai/mli-test1
cd ~/_github/charles-cai/mli-test1
git clone https://github.com/charles-cai/mli-test1.git
```

## Build

```shell
uv venv
source .venv/bin/activate

```

# MNIST Dataset Downloader and Viewer

This project provides utilities for downloading, saving, and inspecting the MNIST dataset from Hugging Face.

## Installation

First, set up your environment and install the required dependencies:

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

