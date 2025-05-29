FROM python:3.12-slim

WORKDIR /app

# Install uv for faster, more reliable dependency management
RUN pip install uv

# Copy requirements and install dependencies using uv
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy model code and API
COPY . .

# Expose port
EXPOSE 8000

# Run the FastAPI API
CMD ["python", "api.py"]
