## Foundation Project

### Overview

Building an end-to-end application on a self-managed server is the best way to prepare for our programme. The foundation project is a simplified version of the projects you will build every week during the programme, so it's an excellent way for you to learn and prepare and for us to assess whether you're ready for our residency. To complete this exercise, you need to build, containerize, and deploy an MNIST digit classifier.

### Project Brief

Goal: Build, containerize, and deploy a simple digit-recogniser trained on the MNIST dataset.

Below is a wireframe of the application you will build. It doesn't need to be laid out exactly as shown, but it should contain all the necessary functionality.

![Project Overview](project_round_one.png)

1. Train a PyTorch Model
 - Develop a basic PyTorch model to classify handwritten digits from the MNIST dataset.
- Train it locally and confirm that it achieves a reasonable accuracy.

2. Interactive Front-End
 - Create a web interface (using Streamlit) where users can draw a digit on a canvas or input area.
 - When the user submits the drawing, the web app should run the trained PyTorch model to produce:
   - Prediction: the model's guess at the digit (0–9).
   - Confidence: the model's probability for its prediction.
   - True Label: allow the user to manually input the correct digit so you can gather feedback.

3. Logging with PostgreSQL
 - Every time a prediction is made, log these details to a PostgreSQL database:
  - Timestamp
  - Predicted digit
  - User-provided true label

4. Containerization with Docker
  - Use Docker to containerize:
    - The PyTorch model/service
    - The Streamlit web app
    - The PostgreSQL database
  - Use Docker Compose to define your multi-container setup in a docker-compose.yml file.

5. Deployment
 - Set up a self-managed server (e.g., Hetzner's basic instance) or any other environment where you can install Docker and control the deployment end-to-end.
 - Deploy your containerized application to the server and make it accessible via a public IP or domain.

6. Add project to GitHub
 - Add your project to GitHub.
 - Make sure to include a README with a link to the live application.
 - Share the link to your GitHub repository with us via the application form.

## Project Structure

This project uses a modular structure:

```
mlx-test1/
├── data/           # MNIST dataset storage
├── model/          # ML model code and training scripts
├── db/             # PostgreSQL schema and setup scripts
├── app/            # Gradio UI (calls API, logs to DB)
├── api/            # FastAPI service (model files copied here for deployment)
└── docker-compose.yml
```

- **/model**: Contains `MnistModel.py` (training, inference) and related files. For deployment, model files are copied to `/api`.
- **/api**: FastAPI service for digit prediction (serves the model in production).
- **/app**: Gradio web interface for drawing and prediction.
- **/db**: PostgreSQL schema and setup scripts.
- **/data**: Downloaded MNIST dataset.

## Setup Instructions

### Environment Setup

```shell
# Clone repository
git clone https://github.com/charles-cai/mlx-test1.git
cd mlx-test1

# Create virtual environment (WSL Ubuntu 22.04 LTS)
uv venv --python=python3.12
source .venv/bin/activate

# Install dependencies for each component
cd model && pip install -r requirements.txt && cd ..
cd app && pip install -r requirements.txt && cd ..

cd api && pip install -r requirements.txt && cd ..
# check build-dockers.sh to copy dependencies from /model to /api
```

### Model Training

```shell
# Train MNIST model
cd model
python MnistModel.py --train --epochs 5
python MnistModel.py --help  # See all available options
cd ..
```

### Database Setup

```shell
# Initialize PostgreSQL schema
cd db
# Follow README.md instructions for database setup
cd ..
```

## Quick Start

```shell
# 1. Clone and setup
git clone https://github.com/charles-cai/mlx-test1.git
cd mlx-test1
uv venv --python=python3.12
source .venv/bin/activate

# 2. Install dependencies
cd model && uv pip install -r requirements.txt && cd ..
cd app && uv pip install -r requirements.txt && cd ..

# 3. Train model (if needed)
cd model
python MnistModel.py --train --epochs 5
cd ..

# 4. Start all services
docker-compose up --build

## Setup Instructions

### Development Workflow

```bash
# 1. Initialize PostgreSQL schema
cd db
./setup-postgresql.sh
cd ..

# 2. Copy model dependencies for API
./build-dockers.sh

# 3. Start the API
./build-dockers.sh
cd api
python api.py
```

## Production Workflow

```bash
# 1. Build Docker images and copy model files
./build-dockers.sh --build

# 2. Start all services
docker compose up
```

# Access points (DEV):
# - Gradio UI: http://localhost:7860
# - FastAPI: http://localhost:8889
# - PostgreSQL: localhost:5432

# Access point (via Cloudflare Tunnel):
(TBC)
