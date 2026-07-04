import os
from pathlib import Path
from datetime import datetime

class Config:
    """Application configuration settings.
    Adjust as needed for production deployment.
    """

    # Base directory of the project
    BASE_DIR = Path(__file__).resolve().parent

    # Secret key for sessions & CSRF (should be overridden via env var)
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')

    # SQLite database URI
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', f"sqlite:///{BASE_DIR / 'database' / 'app.db'}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Logging configuration
    LOG_FOLDER = BASE_DIR / 'logs'
    LOG_FILE = LOG_FOLDER / f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # Uploads and model storage
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MODEL_FOLDER = BASE_DIR / 'models'
    STATIC_IMAGE_FOLDER = BASE_DIR / 'static' / 'images'

    # Misc settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB limit for uploads
    ALLOWED_EXTENSIONS = {'csv'}

    @classmethod
    def init_folders(cls):
        """Create required folders if they do not exist."""
        for folder in [cls.LOG_FOLDER, cls.UPLOAD_FOLDER, cls.MODEL_FOLDER, cls.STATIC_IMAGE_FOLDER, cls.BASE_DIR / 'database']:
            os.makedirs(folder, exist_ok=True)
