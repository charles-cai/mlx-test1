FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy requirements first for better caching
COPY api/requirements.txt ./api_requirements.txt
COPY model/requirements.txt ./model_requirements.txt

# Install Python dependencies using uv
RUN uv pip install --system --no-cache -r api_requirements.txt
RUN uv pip install --system --no-cache -r model_requirements.txt

# Copy the API application code
COPY api/ .

# Create model directory and copy model files
RUN mkdir -p ./model
COPY model/MnistModel.py ./model/
COPY model/*.pth ./model/

# Expose the port the app runs on
EXPOSE 8889

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/model/saved_model.pth

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8889"]