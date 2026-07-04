import os
import logging
import sqlite3
from datetime import datetime

class DBHandler(logging.Handler):
    """
    Custom logging handler that writes log records to the SQLite database.
    Does not use SQLAlchemy to avoid circular imports.
    """
    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path
        
    def emit(self, record):
        try:
            # We connect directly to SQLite to log messages
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Format message
            msg = self.format(record)
            level = record.levelname
            module = record.module
            timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            
            # Insert log
            cursor.execute(
                "INSERT INTO system_logs (timestamp, level, message, module) VALUES (?, ?, ?, ?)",
                (timestamp, level, msg, module)
            )
            conn.commit()
            conn.close()
        except Exception:
            # If database logging fails (e.g. table doesn't exist yet), fall back silently
            pass

def setup_logger(db_path=None):
    logger = logging.getLogger('credit_card_app')
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if setup multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'app.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # DB Handler (if path provided)
    if db_path:
        db_handler = DBHandler(db_path)
        db_handler.setFormatter(formatter)
        logger.addHandler(db_handler)
        
    return logger

# Global default logger (will be updated with DB handler once DB is initialized)
logger = setup_logger()
