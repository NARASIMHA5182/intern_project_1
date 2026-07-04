import os
from flask import Flask, render_template
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from config import Config
from database.models import db, User
from database.db_helper import init_db
from utils.logger import logger

# Import blueprints
from routes.auth_routes import auth_bp
from routes.predict_routes import predict_bp
from routes.admin_routes import admin_bp
from routes.main_routes import main_bp

def create_app():
    # Initialize Flask app
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize directory folders
    Config.init_folders()
    
    # Enable CSRF Protection
    csrf = CSRFProtect(app)
    
    # Initialize Database & seed defaults
    init_db(app)
    
    # Setup Login Manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.login_message_category = 'info'
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
        
    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(main_bp)
    
    # Custom Context Processors
    @app.context_processor
    def inject_system_variables():
        """
        Injects variables like the system name to all pages.
        """
        try:
            from database.models import Setting
            sys_name_setting = Setting.query.filter_by(config_key='system_name').first()
            sys_name = sys_name_setting.config_value if sys_name_setting else "Vanguard Credit"
        except Exception:
            sys_name = "Vanguard Credit"
        return dict(system_name=sys_name)

    # Register Error Handlers
    @app.errorhandler(403)
    def forbidden_error(error):
        logger.warning(f"403 Forbidden trigger: {error}")
        return render_template('errors/403.html'), 403

    @app.errorhandler(404)
    def not_found_error(error):
        logger.info(f"404 Not Found trigger: {error}")
        return render_template('errors/404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"500 Internal Server Error: {error}", exc_info=True)
        db.session.rollback()
        return render_template('errors/500.html'), 500
        
    return app

app = create_app()

if __name__ == '__main__':
    # Start app server
    logger.info("Starting Flask application server...")
    app.run(host='0.0.0.0', port=5000)
