import os
import joblib
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_required, current_user

from database.models import db, User, Prediction
from utils.logger import logger

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def home():
    if current_user.is_authenticated:
        if current_user.is_admin:
            return redirect(url_for('admin.dashboard'))
        return redirect(url_for('predict.predict_form'))
    return render_template('home.html')

@main_bp.route('/about')
def about():
    return render_template('about.html')

@main_bp.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        logger.info(f"Contact query from {name} ({email}): {subject} - {message[:50]}...")
        flash("Thank you for contacting Vanguard Support. Your message has been received.", "success")
        return redirect(url_for('main.contact'))
    return render_template('contact.html')

@main_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        
        # Verify password for changes
        if not current_user.check_password(current_password):
            flash("Incorrect current password.", "danger")
            return redirect(url_for('main.profile'))
            
        try:
            # Change username/email
            if username:
                current_user.username = username
            if email:
                current_user.email = email
                
            # Change password
            if new_password:
                if len(new_password) < 8:
                    flash("New password must be at least 8 characters.", "warning")
                    return redirect(url_for('main.profile'))
                current_user.set_password(new_password)
                logger.info(f"User {current_user.username} updated password.")
                
            db.session.commit()
            flash("Profile updated successfully.", "success")
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating user profile: {str(e)}")
            flash("Error updating profile. Username or email may already be taken.", "danger")
            
        return redirect(url_for('main.profile'))
        
    # Get user's own prediction statistics
    my_preds = Prediction.query.filter_by(user_id=current_user.id).all()
    total_my_preds = len(my_preds)
    approved_my_preds = sum(1 for p in my_preds if p.prediction == 'Approved')
    
    stats = {
        'total': total_my_preds,
        'approved': approved_my_preds,
        'rejected': total_my_preds - approved_my_preds,
    }
    
    return render_template('profile.html', stats=stats)

@main_bp.route('/performance')
@login_required
def performance():
    model_dir = current_app.config['MODEL_FOLDER']
    metrics_path = os.path.join(model_dir, 'metrics.joblib')
    
    metrics = None
    if os.path.exists(metrics_path):
        try:
            metrics = joblib.load(metrics_path)
        except Exception as e:
            logger.error(f"Error loading evaluation metrics: {str(e)}")
            
    if not metrics:
        # Generate metrics on the fly if missing (trigger run_evaluation_pipeline)
        try:
            logger.info("Evaluation metrics file missing on load. Re-running evaluation...")
            from training.trainer import run_training_pipeline
            results, best_run, X_test, y_test = run_training_pipeline()
            
            from evaluation.evaluator import run_evaluation_pipeline
            # Read meta
            meta_path = os.path.join(model_dir, 'model_meta.joblib')
            meta = joblib.load(meta_path)
            
            metrics = run_evaluation_pipeline(meta, best_run['model'], X_test, y_test)
        except Exception as e:
            logger.error(f"Failed to auto-generate metrics: {str(e)}")
            flash("Could not load model performance files.", "warning")
            
    # Gather other comparison models details
    model_name = "Trained Classifier"
    test_auc = 0.0
    try:
        meta_path = os.path.join(model_dir, 'model_meta.joblib')
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            model_name = meta.get('model_name', 'Trained Classifier')
            test_auc = meta.get('test_auc', 0.0)
    except Exception:
        pass
        
    return render_template('performance.html', metrics=metrics, model_name=model_name, test_auc=test_auc)

@main_bp.route('/history')
@login_required
def prediction_history():
    """
    Displays user prediction history table
    """
    if current_user.is_admin:
        predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    else:
        predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.timestamp.desc()).all()
        
    return render_template('history.html', predictions=predictions)

@main_bp.route('/health')
def health():
    """
    Service Health Status Check API
    """
    # Verify DB connectivity
    db_ok = False
    try:
        db.session.execute(db.select(1))
        db_ok = True
    except Exception:
        pass
        
    # Check if pipeline files exist
    pipeline_ok = os.path.exists(os.path.join(current_app.config['MODEL_FOLDER'], 'best_model.joblib'))
    
    status = 'healthy' if (db_ok and pipeline_ok) else 'degraded'
    code = 200 if status == 'healthy' else 500
    
    return {
        'status': status,
        'database': 'connected' if db_ok else 'disconnected',
        'machine_learning_pipeline': 'active' if pipeline_ok else 'inactive'
    }, code
