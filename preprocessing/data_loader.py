import os

# Use the centralised path helper to add the project root to sys.path.
# This allows running this script directly (e.g. python preprocessing/data_loader.py)
# as well as importing it as a module from the project root.
from utils.path_helper import add_project_root
add_project_root()

import numpy as np
import pandas as pd
from utils.logger import logger


def generate_synthetic_dataset(
    output_path: str,
    num_samples: int = 2000,
    seed: int = 42,
) -> None:
    """Generate a realistic synthetic credit dataset and save it as a CSV.

    Produces correlated features (income, credit score, debt ratio, etc.) and
    a logistic-regression-derived target label (``Approved`` / ``Rejected``).
    A small fraction of values are intentionally set to ``NaN`` to exercise the
    data-cleaning pipeline.

    Parameters
    ----------
    output_path : str
        Absolute or relative path where the CSV file will be written.
        Parent directories are created automatically if they do not exist.
    num_samples : int, optional
        Number of synthetic records to generate (default ``2000``).
    seed : int, optional
        Random seed for reproducibility (default ``42``).
    """
    np.random.seed(seed)
    logger.info(f"Generating synthetic credit dataset with {num_samples} records...")
    
    # Age: 21 to 75
    age = np.random.randint(21, 76, size=num_samples)
    
    # Gender: Male, Female, Other
    gender = np.random.choice(['Male', 'Female', 'Other'], size=num_samples, p=[0.48, 0.49, 0.03])
    
    # Education: High School, Bachelor, Master, PhD
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD'], 
        size=num_samples, 
        p=[0.30, 0.50, 0.15, 0.05]
    )
    
    # Occupation & Employment Type
    occupations = ['Software Engineer', 'Manager', 'Accountant', 'Sales Executive', 'Teacher', 'Nurse', 'Self-Employed', 'Student', 'Unemployed', 'Retired']
    employment_types = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed', 'Contract']
    
    occupation_choices = []
    employment_choices = []
    years_employment = []
    
    for a in age:
        if a < 25 and np.random.rand() > 0.6:
            occ = 'Student'
            emp = 'Part-time'
            y = 0.0
        elif a > 65 and np.random.rand() > 0.2:
            occ = 'Retired'
            emp = 'Unemployed'
            y = 0.0
        else:
            occ = np.random.choice(occupations[:-2])
            emp = np.random.choice(employment_types[:3] + ['Contract'], p=[0.70, 0.10, 0.15, 0.05])
            max_y = max(0, a - 22)
            y = round(np.random.uniform(0.5, min(35.0, max_y)), 1)
            
        occupation_choices.append(occ)
        employment_choices.append(emp)
        years_employment.append(y)
        
    occupation = np.array(occupation_choices)
    employment_type = np.array(employment_choices)
    years_of_employment = np.array(years_employment)
    
    # Income (Annual, Monthly, Expenses)
    # Correlated with Education and Occupation
    base_income_map = {
        'High School': 35000,
        'Bachelor': 65000,
        'Master': 90000,
        'PhD': 115000
    }
    
    annual_income = []
    for edu, occ in zip(education, occupation):
        base = base_income_map[edu]
        if occ == 'Software Engineer':
            base += 30000
        elif occ == 'Manager':
            base += 25000
        elif occ == 'Student':
            base = 12000
        elif occ == 'Unemployed':
            base = 5000
        elif occ == 'Retired':
            base = 28000
            
        # Add variance
        income = int(np.random.normal(base, base * 0.15))
        annual_income.append(max(3000, income))
        
    annual_income = np.array(annual_income)
    monthly_income = np.round(annual_income / 12, 2)
    
    # Monthly Expenses: correlated with income
    monthly_expenses = []
    for mi in monthly_income:
        expense_ratio = np.random.uniform(0.25, 0.55)
        # Students / Unemployed spend higher share of their little income
        if mi < 1500:
            expense_ratio = np.random.uniform(0.70, 0.95)
        monthly_expenses.append(round(mi * expense_ratio, 2))
    monthly_expenses = np.array(monthly_expenses)
    
    # Dependents: 0 to 4
    dependents = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.45, 0.25, 0.20, 0.07, 0.03])
    
    # Marital Status: Single, Married, Divorced, Widowed
    marital_status = []
    for a in age:
        if a < 26:
            p = [0.85, 0.13, 0.02, 0.00]
        elif a < 45:
            p = [0.30, 0.55, 0.13, 0.02]
        else:
            p = [0.15, 0.60, 0.20, 0.05]
        marital_status.append(np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], p=p))
    marital_status = np.array(marital_status)
    
    # Residence Type: Owned, Rented, Mortgaged, With Parents
    residence_type = np.random.choice(
        ['Owned', 'Rented', 'Mortgaged', 'With Parents'], 
        size=num_samples, 
        p=[0.35, 0.40, 0.20, 0.05]
    )
    
    # Credit Score: 300 to 850
    credit_score = []
    # Depend on history and income
    for mi in monthly_income:
        base_score = np.random.choice([400, 600, 720], p=[0.15, 0.45, 0.40])
        score = int(np.random.normal(base_score, 70))
        credit_score.append(max(300, min(850, score)))
    credit_score = np.array(credit_score)
    
    # Existing Loans: 0 to 5
    existing_loans = np.random.choice([0, 1, 2, 3, 4, 5], size=num_samples, p=[0.50, 0.25, 0.15, 0.06, 0.03, 0.01])
    
    # Debt Ratio: Total debt service / Income
    # Approximated by monthly debt payments (calculated from existing loans) / monthly income
    debt_ratio = []
    for el, mi, me in zip(existing_loans, monthly_income, monthly_expenses):
        existing_debt_pay = el * np.random.uniform(150, 400)
        total_monthly_obligations = existing_debt_pay + me
        ratio = total_monthly_obligations / mi
        debt_ratio.append(round(min(2.5, ratio), 4))
    debt_ratio = np.array(debt_ratio)
    
    # Bank Balance, Savings, Investment: correlated with income
    bank_balance = []
    savings = []
    investment = []
    for ai, cs in zip(annual_income, credit_score):
        bb = max(50, int(np.random.exponential(ai * 0.15) + np.random.normal(500, 200)))
        sav = max(0, int(bb * np.random.uniform(0.3, 0.8) if bb > 500 else 0))
        inv = max(0, int(ai * np.random.uniform(0, 0.3) * (cs / 850.0)))
        bank_balance.append(bb)
        savings.append(sav)
        investment.append(inv)
        
    bank_balance = np.array(bank_balance)
    savings = np.array(savings)
    investment = np.array(investment)
    
    # Loan Amount Requested: 5k to 100k
    loan_amount = []
    for ai, cs in zip(annual_income, credit_score):
        multiplier = np.random.uniform(0.1, 0.8)
        if cs < 550:
            multiplier *= 0.5
        amt = int(ai * multiplier)
        loan_amount.append(max(2000, min(150000, amt)))
    loan_amount = np.array(loan_amount)
    
    # History indicators: Good, Bad, None
    loan_history = []
    credit_history = []
    for cs in credit_score:
        if cs > 720:
            lh = np.random.choice(['Good', 'None'], p=[0.85, 0.15])
            ch = np.random.choice(['Good', 'None'], p=[0.90, 0.10])
        elif cs > 600:
            lh = np.random.choice(['Good', 'Bad', 'None'], p=[0.50, 0.15, 0.35])
            ch = np.random.choice(['Good', 'Bad', 'None'], p=[0.55, 0.15, 0.30])
        else:
            lh = np.random.choice(['Good', 'Bad', 'None'], p=[0.10, 0.60, 0.30])
            ch = np.random.choice(['Good', 'Bad', 'None'], p=[0.10, 0.70, 0.20])
        loan_history.append(lh)
        credit_history.append(ch)
        
    loan_history = np.array(loan_history)
    credit_history = np.array(credit_history)
    
    # Target variable generation (Approved / Rejected)
    # Using Logistic function with weight parameters for linear factors
    approved_prob = []
    for i in range(num_samples):
        # Base log odds
        score = 0.0
        
        # Credit Score factor (Strongest)
        score += (credit_score[i] - 620) / 45.0
        
        # Debt ratio factor (Negative)
        score -= (debt_ratio[i] - 0.35) * 4.5
        
        # Years of Employment (Positive)
        score += (years_of_employment[i]) * 0.1
        
        # Income ratio factor (Positive)
        score += (annual_income[i] / 50000.0) * 0.5
        
        # Loan requested relative to annual income (Negative if too high)
        loan_inc_ratio = loan_amount[i] / annual_income[i]
        if loan_inc_ratio > 0.5:
            score -= (loan_inc_ratio - 0.5) * 3.0
            
        # Credit History (Very strong)
        if credit_history[i] == 'Good':
            score += 1.5
        elif credit_history[i] == 'Bad':
            score -= 2.5
            
        if loan_history[i] == 'Good':
            score += 0.8
        elif loan_history[i] == 'Bad':
            score -= 1.8
            
        # Savings & Assets (Positive)
        score += (savings[i] + bank_balance[i]) / 40000.0
        
        # Age penalty for very young
        if age[i] < 23:
            score -= 0.5
            
        # Calculate logistic probability
        prob = 1 / (1 + np.exp(-score))
        approved_prob.append(prob)
        
    approved_prob = np.array(approved_prob)
    
    # Apply noise and threshold
    # Add a little noise so it is not 100% deterministic
    prob_with_noise = approved_prob + np.random.normal(0, 0.05, size=num_samples)
    prob_with_noise = np.clip(prob_with_noise, 0.0, 1.0)
    
    approval_status = np.where(prob_with_noise >= 0.50, 'Approved', 'Rejected')
    
    # Save into a Pandas DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Occupation': occupation,
        'Employment Type': employment_type,
        'Education': education,
        'Annual Income': annual_income,
        'Monthly Income': monthly_income,
        'Monthly Expenses': monthly_expenses,
        'Credit Score': credit_score,
        'Loan Amount': loan_amount,
        'Existing Loans': existing_loans,
        'Debt Ratio': debt_ratio,
        'Years of Employment': years_of_employment,
        'Marital Status': marital_status,
        'Residence Type': residence_type,
        'Dependents': dependents,
        'Bank Balance': bank_balance,
        'Savings': savings,
        'Investment': investment,
        'Loan History': loan_history,
        'Credit History': credit_history,
        'Approval Status': approval_status
    })
    
    # Introduce a few random null values to demonstrate cleaning capabilities
    for col in ['Occupation', 'Debt Ratio', 'Years of Employment', 'Bank Balance']:
        mask = np.random.rand(num_samples) < 0.02
        df.loc[mask, col] = np.nan
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset generated and saved successfully to {output_path}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_csv = os.path.join(base_dir, 'dataset', 'credit_card_data.csv')
    generate_synthetic_dataset(output_csv)
