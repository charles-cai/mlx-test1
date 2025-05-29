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
COPY requirements.txt ./

# Install Python dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the application code
COPY . ./

# Expose the port the app runs on
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV DATABASE_URL=postgresql://postgres:password@localhost:5432/mnist_db

# Command to run the application
CMD ["python", "app.py"]