import pandas as pd
import numpy as np
from utils.logger import logger

class DataCleaner:
    """
    Cleans data by handling missing values, duplicates, and outliers.
    """
    def __init__(self):
        self.imputers = {}
        self.bounds = {}
        # Duplicate line removed
        
    def fit(self, df):
        """
        Learns the cleaning parameters (imputation values, etc.) from training data.
        """
        logger.info("Fitting DataCleaner on data...")
        # Handle numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            self.imputers[col] = df[col].median()
            
        # Handle categorical columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if df[col].mode().empty:
                self.imputers[col] = "Unknown"
            else:
                self.imputers[col] = df[col].mode()[0]
        # Compute outlier bounds for numerical columns
        excluded = ["Credit Score", "Age", "Existing Loans", "Dependents", "Years of Employment"]
        for col in numerical_cols:
            if col in excluded:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.bounds[col] = (lower, upper)
        return self

    def transform(self, df):
        """
        Cleans the dataframe based on fitted values.
        """
        df_clean = df.copy()
        
        # 1. Remove duplicate rows (only makes sense if we're not cleaning a single prediction sample)
        if len(df_clean) > 1:
            duplicate_count = df_clean.duplicated().sum()
            if duplicate_count > 0:
                logger.info(f"Removing {duplicate_count} duplicate rows.")
                df_clean = df_clean.drop_duplicates().reset_index(drop=True)
                
        # 2. Impute missing values
        for col in df_clean.columns:
            if col in self.imputers:
                missing_count = df_clean[col].isnull().sum()
                if missing_count > 0:
                    df_clean[col] = df_clean[col].fillna(self.imputers[col])
                    
        # 3. Outlier treatment (Capping outliers with IQR to prevent skewing models)
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col, (lower_bound, upper_bound) in self.bounds.items():
            # Clip values to bounds (standard industry practice)
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            if outliers > 0:
                df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
        return df_clean
        
    def fit_transform(self, df):
        return self.fit(df).transform(df)
