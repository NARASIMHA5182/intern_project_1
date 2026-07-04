from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

from . import db

class User(UserMixin, db.Model):
    """User account for authentication and role management."""
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"

class Prediction(db.Model):
    """Stores each credit‑card approval prediction request and result."""
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    input_data = db.Column(db.JSON, nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    explanation_image = db.Column(db.String(200), nullable=True)  # path to SHAP plot

    user = db.relationship('User', backref=db.backref('predictions', lazy='dynamic'))

    @property
    def age(self):
        return self.input_data.get('Age', self.input_data.get('age'))

    @property
    def gender(self):
        return self.input_data.get('Gender', self.input_data.get('gender'))

    @property
    def occupation(self):
        return self.input_data.get('Occupation', self.input_data.get('occupation'))

    @property
    def employment_type(self):
        return self.input_data.get('Employment Type', self.input_data.get('employment_type'))

    @property
    def education(self):
        return self.input_data.get('Education', self.input_data.get('education'))

    @property
    def annual_income(self):
        return self.input_data.get('Annual Income', self.input_data.get('annual_income'))

    @property
    def monthly_income(self):
        return self.input_data.get('Monthly Income', self.input_data.get('monthly_income'))

    @property
    def monthly_expenses(self):
        return self.input_data.get('Monthly Expenses', self.input_data.get('monthly_expenses'))

    @property
    def credit_score(self):
        return self.input_data.get('Credit Score', self.input_data.get('credit_score'))

    @property
    def loan_amount(self):
        return self.input_data.get('Loan Amount', self.input_data.get('loan_amount'))

    @property
    def existing_loans(self):
        return self.input_data.get('Existing Loans', self.input_data.get('existing_loans'))

    @property
    def debt_ratio(self):
        return self.input_data.get('Debt Ratio', self.input_data.get('debt_ratio'))

    @property
    def years_of_employment(self):
        return self.input_data.get('Years of Employment', self.input_data.get('years_of_employment'))

    @property
    def marital_status(self):
        return self.input_data.get('Marital Status', self.input_data.get('marital_status'))

    @property
    def residence_type(self):
        return self.input_data.get('Residence Type', self.input_data.get('residence_type'))

    @property
    def dependents(self):
        return self.input_data.get('Dependents', self.input_data.get('dependents'))

    @property
    def bank_balance(self):
        return self.input_data.get('Bank Balance', self.input_data.get('bank_balance'))

    @property
    def savings(self):
        return self.input_data.get('Savings', self.input_data.get('savings'))

    @property
    def investment(self):
        return self.input_data.get('Investment', self.input_data.get('investment'))

    @property
    def loan_history(self):
        return self.input_data.get('Loan History', self.input_data.get('loan_history'))

    @property
    def credit_history(self):
        return self.input_data.get('Credit History', self.input_data.get('credit_history'))

    @property
    def approval_status(self):
        return self.prediction

    @property
    def confidence_score(self):
        p = self.probability
        if self.prediction == 'Approved':
            return p * 100.0
        else:
            return (1.0 - p) * 100.0

    @property
    def risk_category(self):
        try:
            from database.models import Setting
            low_t = 0.3
            high_t = 0.7
            low_setting = Setting.query.filter_by(config_key='risk_low_threshold').first()
            if low_setting:
                low_t = float(low_setting.config_value)
            high_setting = Setting.query.filter_by(config_key='risk_high_threshold').first()
            if high_setting:
                high_t = float(high_setting.config_value)
        except Exception:
            low_t = 0.3
            high_t = 0.7

        default_prob = 1.0 - self.probability
        if default_prob <= low_t:
            return 'Low'
        elif default_prob >= high_t:
            return 'High'
        else:
            return 'Medium'

    def __repr__(self):
        return f"<Prediction {self.id} - {self.prediction}>"

class LoginHistory(db.Model):
    """Tracks login attempts for security auditing."""
    __tablename__ = 'login_history'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
    successful = db.Column(db.Boolean, default=False)

    user = db.relationship('User', backref=db.backref('login_events', lazy='dynamic'))

    def __repr__(self):
        return f"<LoginHistory {self.user_id} {'OK' if self.successful else 'FAIL'}>"

class Setting(db.Model):
    """Key‑value store for configurable system settings."""
    __tablename__ = 'settings'
    id = db.Column(db.Integer, primary_key=True)
    config_key = db.Column(db.String(64), unique=True, nullable=False)
    config_value = db.Column(db.String(256), nullable=False)

    def __repr__(self):
        return f"<Setting {self.config_key}={self.config_value}>"

class LogEntry(db.Model):
    """Generic application log stored in the database for admin review."""
    __tablename__ = 'system_logs'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    level = db.Column(db.String(20))
    message = db.Column(db.Text)
    module = db.Column(db.String(64))

    def __repr__(self):
        return f"<LogEntry {self.timestamp} {self.level}>"
