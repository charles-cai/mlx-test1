import logging
from datetime import datetime
from typing import Optional, Any, Dict, List
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class PostgresLogger(Base):
    """SQLAlchemy model for storing ML predictions and labels"""
    __tablename__ = 'postgres_logger'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_name = Column(String(100), nullable=False)
    input_shape = Column(String(50))
    prediction = Column(Float, nullable=False)
    true_label = Column(Float)
    confidence = Column(Float)
    metadata = Column(JSON)
    session_id = Column(String(100))

class PostgresLogger:
    """Logger class for storing ML predictions and labels in PostgreSQL"""
    
    def __init__(self, 
                 database_url: str,
                 model_name: str = "MnistModel",
                 session_id: Optional[str] = None):
        """
        Initialize PostgresLogger
        
        Args:
            database_url: PostgreSQL connection string (e.g., "postgresql://user:password@localhost:5432/dbname")
            model_name: Name of the ML model
            session_id: Optional session identifier for grouping related predictions
        """
        self.database_url = database_url
        self.model_name = model_name
        self.session_id = session_id or str(uuid.uuid4())
        
        # Create engine and session
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create tables if they don't exist
        self.create_tables()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def create_tables(self):
        """Create database tables if they don't exist"""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Error creating database tables: {e}")
            raise
    
    def log_prediction(self,
                      prediction: float,
                      true_label: Optional[float] = None,
                      confidence: Optional[float] = None,
                      input_shape: Optional[str] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log a single prediction to the database
        
        Args:
            prediction: The model's prediction
            true_label: The actual label (if available)
            confidence: Confidence score of the prediction
            input_shape: Shape of the input data
            metadata: Additional metadata as a dictionary
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        session = self.SessionLocal()
        try:
            log_entry = PostgresLogger(
                model_name=self.model_name,
                prediction=float(prediction),
                true_label=float(true_label) if true_label is not None else None,
                confidence=float(confidence) if confidence is not None else None,
                input_shape=input_shape,
                metadata=metadata,
                session_id=self.session_id
            )
            
            session.add(log_entry)
            session.commit()
            self.logger.debug(f"Logged prediction: {prediction}, label: {true_label}")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error logging prediction: {e}")
            return False
        finally:
            session.close()
    
    def log_batch_predictions(self,
                             predictions: List[float],
                             true_labels: Optional[List[float]] = None,
                             confidences: Optional[List[float]] = None,
                             input_shapes: Optional[List[str]] = None,
                             metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """
        Log multiple predictions in a batch
        
        Args:
            predictions: List of model predictions
            true_labels: List of actual labels (if available)
            confidences: List of confidence scores
            input_shapes: List of input shapes
            metadata: List of metadata dictionaries
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        session = self.SessionLocal()
        try:
            batch_entries = []
            
            for i, pred in enumerate(predictions):
                log_entry = PostgresLogger(
                    model_name=self.model_name,
                    prediction=float(pred),
                    true_label=float(true_labels[i]) if true_labels and i < len(true_labels) and true_labels[i] is not None else None,
                    confidence=float(confidences[i]) if confidences and i < len(confidences) and confidences[i] is not None else None,
                    input_shape=input_shapes[i] if input_shapes and i < len(input_shapes) else None,
                    metadata=metadata[i] if metadata and i < len(metadata) else None,
                    session_id=self.session_id
                )
                batch_entries.append(log_entry)
            
            session.add_all(batch_entries)
            session.commit()
            self.logger.info(f"Logged {len(batch_entries)} predictions")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error logging batch predictions: {e}")
            return False
        finally:
            session.close()
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve recent predictions from the database
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of prediction records as dictionaries
        """
        session = self.SessionLocal()
        try:
            results = session.query(PostgresLogger)\
                           .filter_by(model_name=self.model_name)\
                           .order_by(PostgresLogger.timestamp.desc())\
                           .limit(limit)\
                           .all()
            
            return [
                {
                    'id': str(result.id),
                    'timestamp': result.timestamp.isoformat(),
                    'prediction': result.prediction,
                    'true_label': result.true_label,
                    'confidence': result.confidence,
                    'input_shape': result.input_shape,
                    'metadata': result.metadata,
                    'session_id': result.session_id
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"Error retrieving predictions: {e}")
            return []
        finally:
            session.close()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current session
        
        Returns:
            Dictionary with session statistics
        """
        session = self.SessionLocal()
        try:
            from sqlalchemy import func
            
            stats = session.query(
                func.count(PostgresLogger.id).label('total_predictions'),
                func.avg(PostgresLogger.prediction).label('avg_prediction'),
                func.count(PostgresLogger.true_label).label('labeled_count')
            ).filter_by(
                model_name=self.model_name,
                session_id=self.session_id
            ).first()
            
            # Calculate accuracy if we have labels
            accuracy = None
            if stats.labeled_count > 0:
                correct_predictions = session.query(func.count(PostgresLogger.id))\
                    .filter_by(model_name=self.model_name, session_id=self.session_id)\
                    .filter(PostgresLogger.prediction == PostgresLogger.true_label)\
                    .scalar()
                accuracy = correct_predictions / stats.labeled_count if stats.labeled_count > 0 else 0
            
            return {
                'session_id': self.session_id,
                'total_predictions': stats.total_predictions or 0,
                'average_prediction': float(stats.avg_prediction) if stats.avg_prediction else 0.0,
                'labeled_count': stats.labeled_count or 0,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Error getting session stats: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            self.logger.info("Database connections closed")

# Example usage and configuration
if __name__ == "__main__":
    # Example configuration
    DATABASE_URL = "postgresql://username:password@localhost:5432/mlx_test_db"
    
    # Initialize logger
    logger = PostgresLogger(
        database_url=DATABASE_URL,
        model_name="MnistModel",
        session_id="test_session_001"
    )
    
    # Example logging
    logger.log_prediction(
        prediction=7.0,
        true_label=7.0,
        confidence=0.95,
        input_shape="(28, 28, 1)",
        metadata={"epoch": 1, "batch": 10}
    )
    
    # Get recent predictions
    recent = logger.get_recent_predictions(limit=10)
    print(f"Recent predictions: {len(recent)} records")
    
    # Get session statistics
    stats = logger.get_session_stats()
    print(f"Session stats: {stats}")
    
    logger.close()