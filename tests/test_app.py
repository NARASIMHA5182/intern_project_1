import os
import unittest
import tempfile
import pandas as pd
import numpy as np
import json
from flask_login import login_user

from app import create_app
from database.models import db, User, Prediction
from preprocessing.cleaner import DataCleaner
from preprocessing.transformer import DataTransformer

class VanguardSystemTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Create a single database file and configure app class variables once.
        """
        cls.db_fd, cls.db_path = tempfile.mkstemp()
        os.environ['DATABASE_URL'] = f'sqlite:///{cls.db_path}'
        
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['WTF_CSRF_ENABLED'] = False
        cls.client = cls.app.test_client()
        
        with cls.app.app_context():
            db.create_all()

    @classmethod
    def tearDownClass(cls):
        """
        Clean files and environmental variables after class tests finish.
        """
        with cls.app.app_context():
            db.drop_all()
            db.session.remove()
        os.close(cls.db_fd)
        os.unlink(cls.db_path)
        os.environ.pop('DATABASE_URL', None)

    def setUp(self):
        """
        Wipe data tables and seed baseline users to isolate test runs.
        """
        self.client.get('/logout')
        with self.app.app_context():
            db.session.rollback()
            # Clear previous entries to avoid collisions
            Prediction.query.delete()
            User.query.delete()
            db.session.commit()
            
            # Re-seed testing users
            user = User(username='testanalyst', email='test@bank.com', is_admin=False)
            user.set_password('TestPassword123')
            db.session.add(user)
            
            admin = User(username='testadmin', email='testadmin@bank.com', is_admin=True)
            admin.set_password('AdminPassword123')
            db.session.add(admin)
            
            db.session.commit()

    def tearDown(self):
        """
        Rollback session to discard uncommitted items.
        """
        with self.app.app_context():
            db.session.rollback()

    # --- 1. Preprocessing Pipeline Tests ---
    
    def test_data_cleaner(self):
        """
        Verify that missing values are imputed, duplicates are dropped, and outlier boundaries clip properly.
        """
        data = {
            'Age': [30, 30, 30, 30, 30, 45, np.nan],
            'Gender': ['Male', 'Male', 'Male', 'Male', 'Male', 'Female', 'Male'],
            'Annual Income': [50000, 50000, 50000, 50000, 50000, 60000, 1000000] # 1000000 is an outlier
        }
        df = pd.DataFrame(data)
        
        cleaner = DataCleaner()
        df_clean = cleaner.fit_transform(df)
        
        # Deduplication check: duplicate rows should be compressed
        self.assertEqual(len(df_clean), 3)
        # Imputation check: Age median should fill nulls
        self.assertFalse(df_clean['Age'].isnull().any())
        # Outlier check: 1,000,000 should be clipped to Q3 + 1.5 * IQR (which is 62,500)
        self.assertLess(df_clean['Annual Income'].max(), 1000000)


    def test_data_transformer(self):
        """
        Verify that Standard scaling and One-Hot categorical encoding map correctly.
        """
        data = {
            'Gender': ['Male', 'Female', 'Male'],
            'Occupation': ['Engineer', 'Nurse', 'Engineer'],
            'Employment Type': ['Full-time', 'Part-time', 'Full-time'],
            'Education': ['Bachelor', 'High School', 'Bachelor'],
            'Marital Status': ['Single', 'Married', 'Single'],
            'Residence Type': ['Owned', 'Rented', 'Owned'],
            'Loan History': ['Good', 'None', 'Good'],
            'Credit History': ['Good', 'Bad', 'Good'],
            'Age': [25, 40, 35],
            'Annual Income': [50000, 60000, 55000],
            'Monthly Income': [4100, 5000, 4500],
            'Monthly Expenses': [2000, 2500, 2200],
            'Credit Score': [700, 620, 680],
            'Loan Amount': [10000, 15000, 12000],
            'Existing Loans': [1, 2, 1],
            'Debt Ratio': [0.35, 0.45, 0.38],
            'Years of Employment': [2.5, 8.0, 5.5],
            'Dependents': [0, 2, 1],
            'Bank Balance': [2000, 1500, 3000],
            'Savings': [1000, 2000, 1500],
            'Investment': [5000, 0, 8000]
        }
        df = pd.DataFrame(data)
        
        transformer = DataTransformer(scaling_type='standard')
        transformer.fit(df)
        df_trans = transformer.transform(df)
        
        # Verify columns exist
        self.assertIn('Age', df_trans.columns)
        self.assertIn('Gender_Male', df_trans.columns)
        self.assertIn('Gender_Female', df_trans.columns)
        self.assertEqual(len(df_trans), 3)

    # --- 2. Authentication View Tests ---

    def test_login_successful(self):
        """
        Asserts that valid login credentials successfully authenticate the user and redirect to appropriate dashboard.
        """
        response = self.client.post('/login', data={
            'username_or_email': 'testanalyst',
            'password': 'TestPassword123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Should redirect user to the new prediction intake form
        self.assertIn(b'Intake', response.data)

    def test_login_failed(self):
        """
        Asserts that invalid credentials throw login error alerts.
        """
        response = self.client.post('/login', data={
            'username_or_email': 'testanalyst',
            'password': 'WrongPassword999'
        }, follow_redirects=True)
        self.assertIn(b'Invalid username/email', response.data)

    def test_registration_flow(self):
        """
        Verify that new user creation successfully appends records.
        """
        response = self.client.post('/register', data={
            'username': 'newbanker',
            'email': 'new@bank.com',
            'password': 'SecretPassword123',
            'confirm_password': 'SecretPassword123'
        }, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Registration successful', response.data)

    # --- 3. Prediction Engine Tests ---

    def test_unauthorized_redirect(self):
        """
        Verify that unauthorized/anonymous requests to secure endpoints (e.g. prediction form) are blocked.
        """
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 302) # Redirect to login

    def test_predict_form_flow(self):
        """
        Log in as analyst, submit prediction variables, and verify classification metrics are returned.
        """
        # Step 1: Pre-generate model assets first so mock application works!
        from training.trainer import run_training_pipeline
        run_training_pipeline()
        
        # Step 2: Login session
        self.client.post('/login', data={
            'username_or_email': 'testanalyst',
            'password': 'TestPassword123'
        })
        
        # Step 3: Run POST prediction form variables
        response = self.client.post('/predict', data={
            'age': '35',
            'gender': 'Male',
            'occupation': 'Software Engineer',
            'employment_type': 'Full-time',
            'education': 'Bachelor',
            'annual_income': '85000',
            'monthly_income': '7080',
            'monthly_expenses': '2500',
            'credit_score': '740',
            'loan_amount': '25000',
            'existing_loans': '0',
            'debt_ratio': '0.28',
            'years_of_employment': '6.5',
            'marital_status': 'Married',
            'residence_type': 'Mortgaged',
            'dependents': '2',
            'bank_balance': '12000',
            'savings': '32000',
            'investment': '45000',
            'loan_history': 'Good',
            'credit_history': 'Good'
        }, follow_redirects=True)
        
        self.assertEqual(response.status_code, 200)
        # Check if the result details are rendered
        self.assertIn(b'Decision Analysis Report', response.data)
        self.assertIn(b'Approval Prob', response.data)

    # --- 4. System Health Monitor Test ---
    
    def test_health_endpoint(self):
        """
        Asserts that the system health monitoring JSON endpoint reports connected databases.
        """
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['database'], 'connected')

if __name__ == '__main__':
    unittest.main()
