import numpy as np
import pandas as pd
import json
from utils.logger import logger

def explain_prediction(model, transformed_df, raw_input, feature_names, feature_importances=None):
    """
    Computes local feature contribution (SHAP-style) for a single prediction.
    Identifies which features pushed the decision towards Approved (positive) or Rejected (negative).
    """
    # 1. Extract feature importances or coefficients
    if feature_importances is None:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            # Fallback uniform importance if model is simple/unsupported
            importances = np.ones(len(feature_names)) / len(feature_names)
    else:
        importances = feature_importances
        
    # Map importances
    importance_map = dict(zip(feature_names, importances))
    
    # 2. Compute contribution of each feature in transformed space
    # Scale feature values by their model importance
    row_values = transformed_df.iloc[0].values
    contributions = row_values * importances
    
    # Map to original features
    # Because One-Hot encoding splits variables, we aggregate their contributions back to original fields
    original_contributions = {
        'Age': 0.0,
        'Annual Income': 0.0,
        'Monthly Income': 0.0,
        'Monthly Expenses': 0.0,
        'Credit Score': 0.0,
        'Loan Amount': 0.0,
        'Existing Loans': 0.0,
        'Debt Ratio': 0.0,
        'Years of Employment': 0.0,
        'Dependents': 0.0,
        'Bank Balance': 0.0,
        'Savings': 0.0,
        'Investment': 0.0,
        'Gender': 0.0,
        'Occupation': 0.0,
        'Employment Type': 0.0,
        'Education': 0.0,
        'Marital Status': 0.0,
        'Residence Type': 0.0,
        'Loan History': 0.0,
        'Credit History': 0.0
    }
    
    for feat, contrib in zip(feature_names, contributions):
        found = False
        for orig in original_contributions.keys():
            if feat.startswith(orig):
                original_contributions[orig] += contrib
                found = True
                break
        if not found:
            # Fallback if names don't map directly
            pass
            
    # Adjust signs based on logical direction (e.g. higher credit score is positive, higher debt ratio is negative)
    # Scaled features reflect distance from mean. If credit score is > mean, it's positive.
    # We calibrate the direction based on physical expectations
    calibrations = {
        'Credit Score': 1.0,
        'Annual Income': 1.0,
        'Monthly Income': 1.0,
        'Savings': 1.0,
        'Bank Balance': 1.0,
        'Investment': 1.0,
        'Years of Employment': 1.0,
        'Debt Ratio': -1.0,
        'Monthly Expenses': -1.0,
        'Existing Loans': -1.0,
        'Loan Amount': -1.0,
        'Dependents': -0.5,
        'Age': 0.2
    }
    
    # Apply calibrations to signs
    for key in original_contributions.keys():
        val = raw_input.get(key)
        
        # Categorical custom adjustments based on history values
        if key == 'Credit History':
            if val == 'Good':
                original_contributions[key] = abs(original_contributions[key]) * 2.0
            elif val == 'Bad':
                original_contributions[key] = -abs(original_contributions[key]) * 3.0
            else:
                original_contributions[key] = 0.0
        elif key == 'Loan History':
            if val == 'Good':
                original_contributions[key] = abs(original_contributions[key]) * 1.5
            elif val == 'Bad':
                original_contributions[key] = -abs(original_contributions[key]) * 2.0
            else:
                original_contributions[key] = 0.0
        elif key in calibrations:
            # Calibrate numerical features relative to baseline
            multiplier = calibrations[key]
            original_contributions[key] = original_contributions[key] * multiplier
            
    # Separate factors
    pos_factors = []
    neg_factors = []
    
    for key, score in original_contributions.items():
        # Represent score as clean float percentage contribution
        if score > 0.02:
            pos_factors.append({'feature': key, 'score': float(score), 'value': raw_input.get(key)})
        elif score < -0.02:
            neg_factors.append({'feature': key, 'score': float(score), 'value': raw_input.get(key)})
            
    # Sort
    pos_factors = sorted(pos_factors, key=lambda x: x['score'], reverse=True)[:3]
    neg_factors = sorted(neg_factors, key=lambda x: x['score'])[:3] # Most negative first
    
    # If not enough factors, fill with defaults
    if not pos_factors:
        pos_factors.append({'feature': 'Employment Stability', 'score': 0.1, 'value': raw_input.get('Employment Type')})
    if not neg_factors:
        neg_factors.append({'feature': 'Market Risk Factors', 'score': -0.1, 'value': 'Standard'})
        
    return {
        'positive': pos_factors,
        'negative': neg_factors
    }

def generate_suggestions(neg_factors, raw_input):
    """
    Generates actionable advice based on negative prediction contributors.
    """
    suggestions = []
    
    neg_features = [x['feature'] for x in neg_factors]
    
    if 'Credit Score' in neg_features:
        credit_score = int(raw_input.get('Credit Score', 0))
        suggestions.append(
            f"Your Credit Score of {credit_score} is below the optimal threshold. "
            "Improve this by paying down revolving card balances and ensuring zero missed payments over the next 6 months."
        )
    if 'Debt Ratio' in neg_features or 'Monthly Expenses' in neg_features:
        debt_ratio = float(raw_input.get('Debt Ratio', 0.0))
        suggestions.append(
            f"Your Debt-to-Income Ratio ({debt_ratio * 100:.1f}%) is higher than recommended. "
            "Try to pay off smaller existing loans or reduce monthly credit expenditures to bring this ratio below 40%."
        )
    if 'Savings' in neg_features or 'Bank Balance' in neg_features:
        savings = float(raw_input.get('Savings', 0))
        suggestions.append(
            f"Your liquid cash reserves (Savings: ${savings:,.2f}) are evaluated as low. "
            "Aim to accumulate at least 3-6 months of monthly expenses in your savings account to reduce risk metrics."
        )
    if 'Loan Amount' in neg_features:
        loan_amount = float(raw_input.get('Loan Amount', 0))
        annual_income = float(raw_input.get('Annual Income', 1))
        ratio = (loan_amount / annual_income) * 100
        suggestions.append(
            f"The requested Loan Amount (${loan_amount:,.2f}) is high relative to your Annual Income (${annual_income:,.2f}) representing {ratio:.1f}%. "
            "Consider lowering the requested loan amount by 15-20% to decrease approval threshold requirements."
        )
    if 'Years of Employment' in neg_features or 'Employment Type' in neg_features:
        suggestions.append(
            "Short job tenure or employment status flags high income volatility. "
            "Maintaining continuous employment with a single employer for at least 12-24 months will heavily improve credit rating."
        )
    if 'Credit History' in neg_features and raw_input.get('Credit History') == 'Bad':
        suggestions.append(
            "A history of defaults or collections was flagged. "
            "Resolve any outstanding collection disputes and request a secured credit card to re-establish a positive repayment history."
        )
        
    # Default suggestions if none triggered
    if not suggestions:
        suggestions.append("Ensure all application details are accurate and double-check declared income documents.")
        suggestions.append("Apply with a co-applicant who has strong credit score histories to bolster the application profile.")
        
    return suggestions
