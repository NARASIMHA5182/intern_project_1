import os
from database.models import db, User, Setting, LogEntry
from utils.logger import setup_logger, logger

def init_db(app):
    """
    Initialize the database inside the application context.
    Creates tables and seeds default user and system settings.
    """
    with app.app_context():
        db.init_app(app)
        db.create_all()
        
        # Add a logger handler to write log messages into the DB
        db_path = app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', '')
        setup_logger(db_path)
        
        # Seed default settings if empty
        seed_settings()
        
        # Seed default users
        seed_users()
        
        logger.info("Database initialized successfully.")

def seed_settings():
    """
    Seeds default configuration settings in settings table.
    """
    defaults = [
        ('system_name', 'Vanguard Credit Services', 'The name of the banking application shown in the UI.'),
        ('min_credit_score_approved', '650', 'Baseline credit score for auto-approval consideration.'),
        ('max_debt_ratio_approved', '0.45', 'Maximum allowed debt ratio for auto-approval.'),
        ('default_model', 'Best Classifier', 'The current active machine learning model for predictions.'),
        ('risk_low_threshold', '0.3', 'Probability threshold under which application is low risk.'),
        ('risk_high_threshold', '0.7', 'Probability threshold over which application is high risk.')
    ]
    
    for key, val, desc in defaults:
        existing = Setting.query.filter_by(config_key=key).first()
        if not existing:
            new_setting = Setting(config_key=key, config_value=val)
            db.session.add(new_setting)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error seeding settings: {str(e)}")

def seed_users():
    """
    Seeds default admin and user accounts if not present.
    """
    # Create default Admin
    admin_user = User.query.filter_by(username='admin').first()
    if not admin_user:
        admin_user = User(
            username='admin',
            email='admin@bank.com',
            is_admin=True
        )
        admin_user.set_password('AdminPassword123')
        db.session.add(admin_user)
        logger.info("Default admin account created: admin / AdminPassword123")
        
    # Create default Demo User
    demo_user = User.query.filter_by(username='demo').first()
    if not demo_user:
        demo_user = User(
            username='demo',
            email='demo@bank.com',
            is_admin=False
        )
        demo_user.set_password('DemoPassword123')
        db.session.add(demo_user)
        logger.info("Default demo user account created: demo / DemoPassword123")
        
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error seeding users: {str(e)}")
        
def add_system_log(level, message, module=None):
    """
    Utility helper to log messages both into standard logger and Database
    """
    try:
        new_log = LogEntry(level=level, message=message, module=module)
        db.session.add(new_log)
        db.session.commit()
    except Exception:
        db.session.rollback()
