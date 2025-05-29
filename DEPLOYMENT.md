# MNIST Application Deployment Guide

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/charles-cai/mlx-test1.git
cd mlx-test1
uv venv --python=python3.12
source .venv/bin/activate

# 2. Install dependencies
cd model && pip install -r requirements.txt && cd ..
cd app && pip install -r requirements.txt && cd ..

# 3. Train model (if needed)
cd model
python MnistModel.py --train --epochs 5
cd ..

# 4. Start all services
docker-compose up --build

# Access points:
# - Gradio UI: http://localhost:7860
# - FastAPI: http://localhost:8000
# - PostgreSQL: localhost:5432
```

## 📁 Project Structure

```
mlx-test1/
├── data/           # MNIST dataset storage
├── model/          # MnistModel.py + api.py (ML + FastAPI service)
│   └── requirements.txt
├── db/             # PostgreSQL schema initialization and documentation
├── app/            # Gradio UI (communicates with API and database via SQLAlchemy)
│   └── requirements.txt
└── docker-compose.yml
```

## 🔧 Local Development

### Environment Setup
```bash
# Create virtual environment
uv venv --python=python3.12
source .venv/bin/activate

# Install dependencies for each component
cd model && pip install -r requirements.txt && cd ..
cd app && pip install -r requirements.txt && cd ..
```

### Model Training
```bash
cd model
python MnistModel.py --train --epochs 5
python MnistModel.py --help  # See all available options
cd ..
```

### Development Mode

```bash
# Run FastAPI service
cd model
python api.py  # Runs on http://localhost:8000
cd ..

# Run Gradio interface (in another terminal)
cd app
python app.py  # Runs on http://localhost:7860
cd ..
```

### Database Setup
```bash
# Initialize PostgreSQL schema
cd db
# Follow README.md instructions for database setup
cd ..
```

## 🐳 Docker Deployment

### Complete Stack
```bash
# Start all services
docker-compose up --build

# Stop all services
docker-compose down

# View logs
docker-compose logs -f
```

### Individual Services
```bash
# Build and run API service
cd model
docker build -t mnist-api .
docker run -p 8000:8000 mnist-api

# Test the API service
python test_api.py

# Build and run Gradio app
cd app
docker build -t mnist-app .
docker run -p 7860:7860 mnist-app

# PostgreSQL only
docker-compose up postgres
```

### Testing Docker Build
```bash
# Test model service container
cd model
docker build -t mnist-api .
docker run -d --name test-api -p 8000:8000 mnist-api

# Wait for startup and test
sleep 5
chmod +x test.sh
./test.sh

# Clean up test container
docker stop test-api && docker rm test-api
```

## 💾 Database Management

```bash
# Show statistics
python scripts/db_manager.py stats

# Show recent predictions
python scripts/db_manager.py recent --limit 20

# Analyze accuracy by digit
python scripts/db_manager.py accuracy

# Export data to CSV
python scripts/db_manager.py export --filename predictions.csv

# Clear old data (30+ days)
python scripts/db_manager.py clear --days 30
```

## 🌐 Production Deployment

### Environment Variables
```bash
export DATABASE_URL="postgresql://user:pass@host:port/db"
export PYTHONPATH="/app"
```

### Hetzner/VPS Deployment
```bash
# 1. Setup server
ssh root@your-server-ip
apt update && apt install docker.io docker-compose git

# 2. Clone repository
git clone https://github.com/charles-cai/mlx-test1.git
cd mlx-test1

# 3. Deploy
chmod +x start.sh
./start.sh

# 4. Configure firewall
ufw allow 7860  # Gradio
ufw allow 8000  # FastAPI
ufw enable
```

### SSL/Domain Setup
```bash
# Install Caddy for automatic HTTPS
docker run -d \
  --name caddy \
  -p 80:80 -p 443:443 \
  -v $PWD/Caddyfile:/etc/caddy/Caddyfile \
  caddy:alpine

# Caddyfile example:
# your-domain.com {
#     reverse_proxy localhost:7860
# }
# api.your-domain.com {
#     reverse_proxy localhost:8000
# }
```

## 🔍 Monitoring & Maintenance

### Health Checks
```bash
# Check service status
curl http://localhost:8000/health
curl http://localhost:7860

# Database connectivity
python scripts/db_manager.py stats
```

### Logs
```bash
# Docker logs
docker-compose logs -f api
docker-compose logs -f gradio
docker-compose logs -f postgres

# Application logs
tail -f logs/app.log
```

### Backup
```bash
# Database backup
docker exec mnist_postgres pg_dump -U postgres mnist_db > backup.sql

# Model backup
cp model/saved_model.pth backups/model_$(date +%Y%m%d).pth
```

## 🎯 Usage Examples

### Drawing Interface (Gradio)
1. Open http://localhost:7860
2. Draw a digit (0-9) in the canvas
3. Click "Predict Digit"
4. Optionally provide correct label for feedback
5. View confidence scores and database logging

### API Usage
```bash
# Upload image for prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@digit_image.png"

# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## 🚨 Troubleshooting

### Common Issues

**Model not found:**
```bash
cd model
python MnistModel.py --train --epochs 5
```

**Database connection error:**
```bash
docker-compose up postgres
python scripts/db_manager.py create-tables
```

**Port conflicts:**
```bash
# Check what's using ports
netstat -tulpn | grep :7860
netstat -tulpn | grep :8000

# Kill processes
sudo fuser -k 7860/tcp
sudo fuser -k 8000/tcp
```

**Import errors:**
```bash
export PYTHONPATH="/path/to/mlx-test1:$PYTHONPATH"
```

### Reset Everything
```bash
# Stop all containers
docker-compose down -v

# Remove all data
docker system prune -a
rm -rf postgres_data logs

# Start fresh
./start.sh
```