import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from utils.logger import logger

class DataTransformer:
    """
    Transforms clean raw features into machine-learning ready datasets.
    Handles encoding, scaling, column alignment, and data balancing.
    """
    def __init__(self, scaling_type='standard'):
        self.scaling_type = scaling_type
        self.encoder = None
        self.scaler = None
        self.categorical_cols = [
            'Gender', 'Occupation', 'Employment Type', 'Education', 
            'Marital Status', 'Residence Type', 'Loan History', 'Credit History'
        ]
        self.numerical_cols = [
            'Age', 'Annual Income', 'Monthly Income', 'Monthly Expenses', 
            'Credit Score', 'Loan Amount', 'Existing Loans', 'Debt Ratio', 
            'Years of Employment', 'Dependents', 'Bank Balance', 'Savings', 'Investment'
        ]
        self.feature_columns = [] # Order of columns after encoding
        
    def fit(self, df):
        """
        Fits encoder and scaler on training data.
        """
        logger.info("Fitting DataTransformer...")
        # 1. Fit OneHotEncoder on categorical columns
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.encoder.fit(df[self.categorical_cols])
        
        # Determine the feature column names after encoding
        encoded_cat_names = self.encoder.get_feature_names_out(self.categorical_cols)
        self.feature_columns = list(self.numerical_cols) + list(encoded_cat_names)
        
        # 2. Fit Scaler on numerical columns
        if self.scaling_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
            
        # Fit scaler on numerical data
        self.scaler.fit(df[self.numerical_cols])
        return self
        
    def transform(self, df):
        """
        Applies scaling and encoding to input dataframe.
        """
        # Ensure we have all columns
        df_temp = df.copy()
        
        # Scale numeric features
        numerical_scaled = self.scaler.transform(df_temp[self.numerical_cols])
        df_num = pd.DataFrame(numerical_scaled, columns=self.numerical_cols, index=df_temp.index)
        
        # Encode categorical features
        cat_encoded = self.encoder.transform(df_temp[self.categorical_cols])
        encoded_cat_names = self.encoder.get_feature_names_out(self.categorical_cols)
        df_cat = pd.DataFrame(cat_encoded, columns=encoded_cat_names, index=df_temp.index)
        
        # Combine
        X_trans = pd.concat([df_num, df_cat], axis=1)
        
        # Reorder to match trained feature column order
        X_trans = X_trans[self.feature_columns]
        return X_trans
        
    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def prepare_dataset(self, df, target_col='Approval Status', test_size=0.2, balance_data=True, seed=42):
        """
        Complete pipeline to split and process data.
        Returns X_train, X_test, y_train, y_test
        """
        # Extract features and target
        X = df.drop(columns=[target_col])
        # Convert target to binary: Approved -> 1, Rejected -> 0
        y = df[target_col].map({'Approved': 1, 'Rejected': 0}).fillna(0).astype(int).values
        
        # Train test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
        
        # Fit transformer on training set and transform both
        self.fit(X_train_raw)
        X_train = self.transform(X_train_raw)
        X_test = self.transform(X_test_raw)
        
        # Balance training data if required
        if balance_data:
            X_train, y_train = self._balance_dataset(X_train, y_train, seed)
            
        return X_train, X_test, y_train, y_test

    def _balance_dataset(self, X, y, seed=42):
        """
        Performs a native Random Oversampling to balance classes.
        Avoids external dependencies like imbalanced-learn which may fail to install.
        """
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            return X, y
            
        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        
        minority_idx = np.where(y == minority_class)[0]
        majority_idx = np.where(y == majority_class)[0]
        
        diff = len(majority_idx) - len(minority_idx)
        if diff <= 0:
            return X, y
            
        logger.info(f"Imbalance detected. Balancing dataset by oversampling minority class ({diff} instances)...")
        
        np.random.seed(seed)
        oversampled_minority_idx = np.random.choice(minority_idx, size=diff, replace=True)
        
        # Combine indices
        new_indices = np.concatenate([np.arange(len(y)), oversampled_minority_idx])
        
        # Extract new features and target
        if isinstance(X, pd.DataFrame):
            X_balanced = X.iloc[new_indices].reset_index(drop=True)
        else:
            X_balanced = X[new_indices]
            
        y_balanced = y[new_indices]
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_balanced))
        if isinstance(X_balanced, pd.DataFrame):
            X_balanced = X_balanced.iloc[shuffle_idx].reset_index(drop=True)
        else:
            X_balanced = X_balanced[shuffle_idx]
        y_balanced = y_balanced[shuffle_idx]
        
        return X_balanced, y_balanced
