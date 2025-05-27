# PostgreSQL Docker Setup for MNIST

Simple PostgreSQL database setup using Docker for the MNIST prediction project.

## Quick Setup

```bash
cd db
chmod +x setup.sh
./setup.sh
```

## Files

- `init.sql` - Database schema and table definitions
- `setup.sh` - Docker setup script for PostgreSQL 17

## Database Schema

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    predicted_digit INTEGER NOT NULL,
    confidence FLOAT NOT NULL,
    label INTEGER,  -- Nullable: for manual correction of prediction errors
    session_id VARCHAR(255)
);
```

## Connection Details

- **Container**: mnist_postgres  
- **Host**: localhost:5432
- **Database**: mnist_db
- **User**: postgres
- **Password**: password

## Usage

### Setup and Management
```bash
# Full setup (creates container + loads schema)
./setup.sh

# Load schema only (assumes container is running)
./setup.sh --init

# Connect to database
docker exec -it mnist_postgres psql -U postgres -d mnist_db

# Start/stop container
docker start mnist_postgres
docker stop mnist_postgres
```

### Basic Queries
```sql
-- View tables
\dt

-- Recent predictions
SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5;

-- Count predictions by digit
SELECT predicted_digit, COUNT(*) FROM predictions GROUP BY predicted_digit;

-- Predictions with manual labels (error corrections)
SELECT * FROM predictions WHERE label IS NOT NULL;
```

## Troubleshooting

```bash
# Check container status
docker ps | grep mnist_postgres

# View logs
docker logs mnist_postgres

# Reset database
docker stop mnist_postgres
docker rm mnist_postgres
./setup.sh
```