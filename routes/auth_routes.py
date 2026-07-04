from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError

from database.models import db, User, LoginHistory
from utils.logger import logger

auth_bp = Blueprint('auth', __name__)

# --- WTForms Definition ---

class LoginForm(FlaskForm):
    username_or_email = StringField('Username or Email', validators=[DataRequired(), Length(min=3, max=120)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField('Password', validators=[
        DataRequired(), 
        Length(min=8, message="Password must be at least 8 characters long.")
    ])
    confirm_password = PasswordField('Confirm Password', validators=[
        DataRequired(), 
        EqualTo('password', message="Passwords must match.")
    ])
    submit = SubmitField('Create Account')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('Username is already taken. Please choose another one.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('Email is already registered. Please login or use a different email.')

# --- Authentication Routes ---

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('admin.dashboard') if current_user.is_admin else url_for('main.home'))

        
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter(
            (User.username == form.username_or_email.data) | 
            (User.email == form.username_or_email.data)
        ).first()
        
        ip_addr = request.remote_addr
        
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            
            # Record Successful Login
            history = LoginHistory(user_id=user.id, ip_address=ip_addr, successful=True)
            db.session.add(history)
            db.session.commit()
            
            logger.info(f"User {user.username} logged in successfully from IP {ip_addr}.")
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('admin.dashboard') if user.is_admin else url_for('predict.predict_form'))
        else:
            # Record Failed Login (if user exists)
            if user:
                history = LoginHistory(user_id=user.id, ip_address=ip_addr, successful=False)
                db.session.add(history)
                db.session.commit()
            
            logger.warning(f"Failed login attempt for {form.username_or_email.data} from IP {ip_addr}.")
            flash('Invalid username/email or password.', 'danger')
            
    return render_template('login.html', form=form)

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))
        
    form = RegisterForm()
    if form.validate_on_submit():
        new_user = User(
            username=form.username.data,
            email=form.email.data,
            is_admin=False
        )
        new_user.set_password(form.password.data)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            
            logger.info(f"New user registered: {new_user.username} (Email: {new_user.email})")
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during registration: {str(e)}")
            flash('An error occurred during registration. Please try again.', 'danger')
            
    return render_template('register.html', form=form)

@auth_bp.route('/logout')
@login_required
def logout():
    username = current_user.username
    logout_user()
    logger.info(f"User {username} logged out.")
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            logger.info(f"Password reset link requested for email: {email}")
            # In production, send email reset token. Here, mock it:
            flash('A secure password reset link has been dispatched to your email address.', 'success')
        else:
            # Prevent user enumeration
            flash('A secure password reset link has been dispatched to your email address.', 'success')
        return redirect(url_for('auth.login'))
    return render_template('forgot_password.html')
