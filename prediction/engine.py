import os
import joblib
import matplotlib
matplotlib.use('Agg') # Ensure non-interactive backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd

from utils.helpers import get_project_root
from utils.explainers import explain_prediction

MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'best_model.joblib'
CLEANER_PATH = Path(__file__).resolve().parents[1] / 'models' / 'data_cleaner.joblib'
TRANSFORMER_PATH = Path(__file__).resolve().parents[1] / 'models' / 'data_transformer.joblib'
SHAP_DIR = Path(__file__).resolve().parents[1] / 'static' / 'images' / 'shap'
os.makedirs(SHAP_DIR, exist_ok=True)

KEY_MAP = {
    'age': 'Age',
    'gender': 'Gender',
    'occupation': 'Occupation',
    'employment_type': 'Employment Type',
    'education': 'Education',
    'annual_income': 'Annual Income',
    'monthly_income': 'Monthly Income',
    'monthly_expenses': 'Monthly Expenses',
    'credit_score': 'Credit Score',
    'loan_amount': 'Loan Amount',
    'existing_loans': 'Existing Loans',
    'debt_ratio': 'Debt Ratio',
    'years_of_employment': 'Years of Employment',
    'marital_status': 'Marital Status',
    'residence_type': 'Residence Type',
    'dependents': 'Dependents',
    'bank_balance': 'Bank Balance',
    'savings': 'Savings',
    'investment': 'Investment',
    'loan_history': 'Loan History',
    'credit_history': 'Credit History'
}

def clean_input_types(input_dict: dict) -> dict:
    mapped = {}
    for k, v in input_dict.items():
        title_k = KEY_MAP.get(k.lower(), k)
        mapped[title_k] = v
        
    # Convert numerical values
    num_cols = [
        'Age', 'Annual Income', 'Monthly Income', 'Monthly Expenses', 
        'Credit Score', 'Loan Amount', 'Existing Loans', 'Debt Ratio', 
        'Years of Employment', 'Dependents', 'Bank Balance', 'Savings', 'Investment'
    ]
    int_cols = ['Age', 'Credit Score', 'Existing Loans', 'Dependents']
    
    for col in num_cols:
        if col in mapped:
            try:
                val_str = str(mapped[col]).strip()
                if val_str == '' or val_str.lower() == 'nan':
                    mapped[col] = np.nan
                elif col in int_cols:
                    mapped[col] = int(float(val_str))
                else:
                    mapped[col] = float(val_str)
            except (ValueError, TypeError):
                pass
    return mapped

def load_assets():
    """Load the serialized preprocessors and model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not CLEANER_PATH.exists():
        raise FileNotFoundError(f"Cleaner file not found at {CLEANER_PATH}")
    if not TRANSFORMER_PATH.exists():
        raise FileNotFoundError(f"Transformer file not found at {TRANSFORMER_PATH}")
    
    model = joblib.load(MODEL_PATH)
    cleaner = joblib.load(CLEANER_PATH)
    transformer = joblib.load(TRANSFORMER_PATH)
    return model, cleaner, transformer

def predict(input_dict: dict) -> dict:
    """Run prediction on a single input record.

    Args:
        input_dict: Mapping of feature name to value.
    Returns:
        dict with keys: prediction (str), probability (float), shap_image (str path).
    """
    model, cleaner, transformer = load_assets()
    
    # Clean and align types and names
    mapped = clean_input_types(input_dict)
    
    # Convert to single-row DataFrame
    df = pd.DataFrame([mapped])
    
    # Process through pipeline
    df_clean = cleaner.transform(df)
    X_trans = transformer.transform(df_clean)
    
    # Get prediction probability
    prob = model.predict_proba(X_trans)[0, 1]  # probability of positive class (Approved)
    pred_label = 'Approved' if prob >= 0.5 else 'Rejected'
    
    # Generate explanations
    explanation = explain_prediction(model, X_trans, mapped, transformer.feature_columns)
    
    # Plot feature contributions horizontal bar chart
    features = []
    scores = []
    colors = []
    
    # Combine positive and negative drivers
    drivers = explanation['positive'] + explanation['negative']
    drivers = sorted(drivers, key=lambda x: abs(x['score']))
    
    for d in drivers:
        features.append(f"{d['feature']} ({d['value']})")
        scores.append(d['score'])
        colors.append('#10b981' if d['score'] > 0 else '#ef4444')
        
    plt.figure(figsize=(8, 4))
    plt.barh(features, scores, color=colors)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.title('Decision Influencing Factors (Contribution Scores)')
    plt.xlabel('Contribution Direction & Strength')
    plt.tight_layout()
    
    shap_path = SHAP_DIR / f"shap_{int(np.random.randint(1e6))}.png"
    plt.savefig(shap_path)
    plt.close()
    
    return {
        'prediction': pred_label,
        'probability': float(prob),
        'shap_image': str(shap_path.relative_to(get_project_root()))
    }
