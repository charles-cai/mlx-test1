-- Initialize MNIST database
CREATE DATABASE IF NOT EXISTS mnist_db;

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    predicted_digit INTEGER NOT NULL,
    confidence FLOAT NOT NULL,
    label INTEGER,
    session_id VARCHAR(255)
);