"""
Database connection and data ingestion from real sources.
"""
import pandas as pd
from typing import Optional, Dict, List
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseConnection:
    """Manage database connections and queries."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database connection.
        
        Args:
            connection_string: Database URL (uses settings if not provided)
        """
        self.connection_string = connection_string or settings.DATABASE_URL
        self.engine: Optional[Engine] = None
    
    def connect(self) -> Engine:
        """Establish database connection."""
        try:
            if not self.connection_string:
                raise ValueError("Database connection string not configured")
            
            self.engine = create_engine(self.connection_string, pool_pre_ping=True)
            logger.info("Database connection established")
            return self.engine
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            Query results as DataFrame
        """
        try:
            if not self.engine:
                self.connect()
            
            logger.info(f"Executing query: {query[:100]}...")
            df = pd.read_sql(text(query), self.engine, params=params)
            logger.info(f"Query returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class DataIngestion:
    """Ingest data from various sources (database, API, files)."""
    
    def __init__(self):
        self.db = DatabaseConnection()
    
    def fetch_users(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch user data from database.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            User DataFrame or None if error
        """
        try:
            query = """
                SELECT user_id, name, email, age, gender, city, state,
                       device_type, membership, signup_date
                FROM users
                WHERE signup_date BETWEEN :start_date AND :end_date
            """
            
            df = self.db.execute_query(
                query,
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            logger.info(f"Fetched {len(df)} users")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch users: {e}")
            return None
    
    def fetch_orders(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch order data from database.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Orders DataFrame or None if error
        """
        try:
            query = """
                SELECT order_id, user_id, order_date, total_amount,
                       status, payment_method, discount_pct
                FROM orders
                WHERE order_date BETWEEN :start_date AND :end_date
            """
            
            df = self.db.execute_query(
                query,
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            logger.info(f"Fetched {len(df)} orders")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch orders: {e}")
            return None
    
    def fetch_sessions(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch session data from database."""
        try:
            query = """
                SELECT session_id, user_id, session_start, session_end,
                       duration_secs, device, referral_source, pages_visited, bounced
                FROM sessions
                WHERE session_start BETWEEN :start_date AND :end_date
            """
            
            df = self.db.execute_query(
                query,
                params={'start_date': start_date, 'end_date': end_date}
            )
            
            logger.info(f"Fetched {len(df)} sessions")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch sessions: {e}")
            return None
    
    def fetch_all(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch all data sources.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dictionary of DataFrames
        """
        logger.info(f"Fetching all data from {start_date} to {end_date}")
        
        data = {}
        
        # Fetch each table
        tables = {
            'users': self.fetch_users,
            'orders': self.fetch_orders,
            'sessions': self.fetch_sessions,
        }
        
        for name, fetch_func in tables.items():
            df = fetch_func(start_date, end_date)
            if df is not None:
                data[name] = df
            else:
                logger.warning(f"Failed to fetch {name}")
        
        logger.info(f"Successfully fetched {len(data)} tables")
        return data
    
    def load_from_csv_fallback(self) -> Dict[str, pd.DataFrame]:
        """
        Fallback: Load data from CSV files if database unavailable.
        
        Returns:
            Dictionary of DataFrames
        """
        logger.warning("Using CSV fallback - loading from files")
        
        data = {}
        csv_files = ['users', 'orders', 'sessions', 'products', 'browse_events', 'order_items']
        
        for name in csv_files:
            try:
                file_path = settings.RAW_DATA_DIR / f"{name}.csv"
                if file_path.exists():
                    data[name] = pd.read_csv(file_path)
                    logger.info(f"Loaded {name} from CSV: {len(data[name])} rows")
            except Exception as e:
                logger.error(f"Failed to load {name} from CSV: {e}")
        
        return data
