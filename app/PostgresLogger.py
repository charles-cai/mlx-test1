import logging
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

Base = declarative_base()

class PredictionRecord(Base):
    """SQLAlchemy model for MNIST predictions table"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    predicted_digit = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    label = Column(Integer, nullable=True)  # For manual corrections

class PostgresLogger:
    """Logger class for storing MNIST predictions in PostgreSQL"""
    
    def __init__(self, 
                 database_url: str,
                 model_name: str = "MnistModel"):
        """
        Initialize PostgresLogger
        
        Args:
            database_url: PostgreSQL connection string
            model_name: Name of the ML model (for compatibility)
        """
        self.database_url = database_url
        self.model_name = model_name
        
        # Create engine and session
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self,
                      prediction: float,
                      true_label: Optional[float] = None,
                      confidence: Optional[float] = None,
                      input_shape: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> bool:
        """
        Log a single prediction to the database
        
        Args:
            prediction: The predicted digit (0-9)
            true_label: The actual digit (if available, for manual corrections)
            confidence: Confidence score of the prediction
            input_shape: Not used in simplified schema (kept for compatibility)
            metadata: Not used in simplified schema (kept for compatibility)
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        session = self.SessionLocal()
        try:
            log_entry = PredictionRecord(
                predicted_digit=int(prediction),
                confidence=float(confidence) if confidence is not None else 0.0,
                label=int(true_label) if true_label is not None else None
            )
            
            session.add(log_entry)
            session.commit()
            self.logger.debug(f"Logged prediction: {prediction}, confidence: {confidence}")
            return True
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error logging prediction: {e}")
            return False
        finally:
            session.close()
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict]:
        """
        Retrieve recent predictions from the database (latest timestamp first)
        
        Args:
            limit: Maximum number of records to retrieve (default 100)
            
        Returns:
            List of prediction records as dictionaries
        """
        session = self.SessionLocal()
        try:
            results = session.query(PredictionRecord)\
                           .order_by(PredictionRecord.created_at.desc())\
                           .limit(limit)\
                           .all()
            
            return [
                {
                    'id': result.id,
                    'created_at': result.created_at.isoformat(),
                    'predicted_digit': result.predicted_digit,
                    'confidence': result.confidence,
                    'label': result.label
                }
                for result in results
            ]
            
        except Exception as e:
            self.logger.error(f"Error retrieving predictions: {e}")
            return []
        finally:
            session.close()
    
    def get_session_stats(self) -> Dict:
        """
        Get statistics for all predictions
        
        Returns:
            Dictionary with overall statistics
        """
        session = self.SessionLocal()
        try:
            from sqlalchemy import func
            
            # Get overall stats
            stats = session.query(
                func.count(PredictionRecord.id).label('total_predictions'),
                func.avg(PredictionRecord.predicted_digit).label('avg_predicted_digit'),
                func.count(PredictionRecord.label).label('labeled_count')
            ).first()
            
            # Calculate accuracy if we have labels
            accuracy = None
            if stats.labeled_count > 0:
                correct_predictions = session.query(func.count(PredictionRecord.id))\
                    .filter(PredictionRecord.predicted_digit == PredictionRecord.label)\
                    .scalar()
                accuracy = correct_predictions / stats.labeled_count if stats.labeled_count > 0 else 0
            
            return {
                'total_predictions': stats.total_predictions or 0,
                'average_predicted_digit': float(stats.avg_predicted_digit) if stats.avg_predicted_digit else 0.0,
                'labeled_count': stats.labeled_count or 0,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
        finally:
            session.close()
    
    def close(self):
        """Close database connections"""
        if hasattr(self, 'engine'):
            self.engine.dispose()
            self.logger.info("Database connections closed")

# Example usage
if __name__ == "__main__":
    # Example configuration
    DATABASE_URL = "postgresql://postgres:password@localhost:5432/mnist_db"
    
    # Initialize logger
    logger = PostgresLogger(
        database_url=DATABASE_URL,
        session_id="test_session_001"
    )
    
    # Example logging
    logger.log_prediction(
        prediction=7,
        confidence=0.95
    )
    
    # Get recent predictions
    recent = logger.get_recent_predictions(limit=10)
    print(f"Recent predictions: {len(recent)} records")
    
    # Get session statistics
    stats = logger.get_session_stats()
    print(f"Session stats: {stats}")
    
    logger.close()