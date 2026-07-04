from flask import Blueprint, render_template, redirect, url_for, flash, request, abort, current_app
from flask_login import login_required, current_user
from functools import wraps
import json
import os

from database.models import db, User, Prediction, LogEntry, Setting, LoginHistory
from utils.logger import logger

admin_bp = Blueprint('admin', __name__)

def admin_required(f):
    """
    Decorator to restrict route access to administrators.
    """
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            logger.warning(f"Unauthorized admin panel access attempt by user: {current_user.username}")
            abort(403) # Forbidden
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/admin/dashboard')
@admin_required
def dashboard():
    # 1. Gather KPIs
    total_users = User.query.count()
    total_preds = Prediction.query.count()
    
    approved_count = Prediction.query.filter_by(prediction='Approved').count()
    rejected_count = Prediction.query.filter_by(prediction='Rejected').count()
    
    approval_rate = (approved_count / total_preds * 100) if total_preds > 0 else 0.0
    
    # Average Credit Score (not stored — omit)
    avg_credit_score = 0.0
    
    # Recent Predictions (last 15)
    recent_predictions = Prediction.query.order_by(Prediction.timestamp.desc()).limit(15).all()
    
    # Recent Logins (last 10)
    recent_logins = LoginHistory.query.order_by(LoginHistory.timestamp.desc()).limit(10).all()
    
    # Model info from meta file
    model_name = "Unknown"
    test_auc = 0.0
    try:
        model_dir = current_app.config['MODEL_FOLDER']
        meta_path = os.path.join(model_dir, 'model_meta.joblib')
        if os.path.exists(meta_path):
            import joblib
            meta = joblib.load(meta_path)
            model_name = meta.get('model_name', 'Trained Classifier')
            test_auc = meta.get('test_auc', 0.0)
    except Exception:
        pass
        
    stats = {
        'total_users': total_users,
        'total_predictions': total_preds,
        'approved_count': approved_count,
        'rejected_count': rejected_count,
        'approval_rate': round(approval_rate, 2),
        'avg_credit_score': avg_credit_score,
        'model_name': model_name,
        'model_auc': round(test_auc * 100, 2)
    }
    
    return render_template('dashboard.html', stats=stats, recent_predictions=recent_predictions, recent_logins=recent_logins)

@admin_bp.route('/admin/logs')
@admin_required
def view_logs():
    # Fetch log entries
    logs = LogEntry.query.order_by(LogEntry.timestamp.desc()).limit(100).all()
    return render_template('admin_logs.html', logs=logs)

@admin_bp.route('/admin/settings', methods=['GET', 'POST'])
@admin_required
def system_settings():
    if request.method == 'POST':
        try:
            for key, val in request.form.items():
                setting = Setting.query.filter_by(config_key=key).first()
                if setting:
                    setting.config_value = val
                    logger.info(f"System setting updated: {key} -> {val}")
            db.session.commit()
            flash("System configurations updated successfully.", "success")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to update settings: {str(e)}")
            flash("Error updating system settings.", "danger")
            
        return redirect(url_for('admin.system_settings'))
        
    # GET - Render settings list
    settings_list = Setting.query.all()
    return render_template('admin_settings.html', settings=settings_list)

@admin_bp.route('/admin/users')
@admin_required
def manage_users():
    users_list = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users_list)
