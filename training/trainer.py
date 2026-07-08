import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    ExtraTreesClassifier
)
from sklearn.metrics import accuracy_score, roc_auc_score

from utils.logger import logger
from preprocessing.cleaner import DataCleaner
from preprocessing.transformer import DataTransformer

# Import advanced classifiers conditionally
XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost is not installed. Skipping XGBoost in model comparison.")

LGBM_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    logger.warning("LightGBM is not installed. Skipping LightGBM in model comparison.")

CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost is not installed. Skipping CatBoost in model comparison.")


class ModelTrainer:
    """
    Manages the training, tuning, comparison, and serialization of multiple models.
    """
    def __init__(self, model_dir=None):
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'
        )
        os.makedirs(self.model_dir, exist_ok=True)
        self.best_model_name = None
        self.best_model = None
        self.best_score = 0.0
        
    def get_candidate_models(self):
        """
        Returns dictionaries of models and their tuning parameters.
        """
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Extra Trees': ExtraTreesClassifier(random_state=42)
        }
        
        param_grids = {
            'Logistic Regression': {'C': [0.1, 1.0, 10.0]},
            'Decision Tree': {'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
            'Random Forest': {'n_estimators': [50, 100], 'max_depth': [6, 10, None]},
            'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1]},
            'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]},
            'Extra Trees': {'n_estimators': [50, 100], 'max_depth': [6, 10, None]}
        }
        
        if XGB_AVAILABLE:
            models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            param_grids['XGBoost'] = {'n_estimators': [50, 100], 'max_depth': [3, 6], 'learning_rate': [0.05, 0.1]}
            
        if LGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1)
            param_grids['LightGBM'] = {'n_estimators': [50, 100], 'max_depth': [3, 6], 'learning_rate': [0.05, 0.1]}
            
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = CatBoostClassifier(random_state=42, verbose=0)
            param_grids['CatBoost'] = {'iterations': [100], 'depth': [4, 6], 'learning_rate': [0.05, 0.1]}
            
        return models, param_grids

    def train_and_compare(self, X_train, y_train, X_test, y_test):
        """
        Trains and compares all candidate models using GridSearchCV.
        Selects the best performing model based on test ROC AUC score.
        """
        logger.info("Starting model training and comparison...")
        models, param_grids = self.get_candidate_models()
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        results = []
        
        for name, model in models.items():
            logger.info(f"Tuning and training {name}...")
            grid = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1
            )
            try:
                grid.fit(X_train, y_train)
                best_tuned = grid.best_estimator_
                
                # Evaluate on test set
                y_pred = best_tuned.predict(X_test)
                y_prob = best_tuned.predict_proba(X_test)[:, 1] if hasattr(best_tuned, "predict_proba") else y_pred
                
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)
                
                logger.info(f"Model: {name} | Best Grid CV AUC: {grid.best_score_:.4f} | Test AUC: {auc:.4f} | Test Acc: {acc:.4f}")
                
                results.append({
                    'name': name,
                    'model': best_tuned,
                    'test_auc': auc,
                    'test_accuracy': acc,
                    'cv_score': grid.best_score_
                })
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                
        # Select best model based on Test AUC
        if not results:
            raise ValueError("No models trained successfully.")
            
        best_run = max(results, key=lambda x: x['test_auc'])
        self.best_model_name = best_run['name']
        self.best_model = best_run['model']
        self.best_score = best_run['test_auc']
        
        logger.info(f"*** Best Model Selected: {self.best_model_name} with Test AUC: {self.best_score:.4f} ***")
        return results, best_run

    def save_pipeline(self, cleaner, transformer):
        """
        Saves the best model, data cleaner, and data transformer to disk.
        """
        logger.info("Saving trained pipeline assets to models directory...")
        
        # Save preprocessors
        cleaner_path = os.path.join(self.model_dir, 'data_cleaner.joblib')
        transformer_path = os.path.join(self.model_dir, 'data_transformer.joblib')
        model_path = os.path.join(self.model_dir, 'best_model.joblib')
        meta_path = os.path.join(self.model_dir, 'model_meta.joblib')
        
        joblib.dump(cleaner, cleaner_path)
        joblib.dump(transformer, transformer_path)
        joblib.dump(self.best_model, model_path)
        
        meta = {
            'model_name': self.best_model_name,
            'test_auc': self.best_score,
            'features': transformer.feature_columns
        }
        joblib.dump(meta, meta_path)
        logger.info("Pipeline assets successfully serialized.")

def run_training_pipeline():
    """
    Full pipeline run: load, clean, transform, train, evaluate, and save.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'dataset', 'credit_card_data.csv')
    
    # 1. Load Data
    if not os.path.exists(dataset_path):
        from preprocessing.data_loader import generate_synthetic_dataset
        generate_synthetic_dataset(dataset_path)
        
    df = pd.read_csv(dataset_path)
    
    # 2. Clean Data
    cleaner = DataCleaner()
    df_clean = cleaner.fit_transform(df)
    
    # 3. Transform and Split
    transformer = DataTransformer(scaling_type='standard')
    X_train, X_test, y_train, y_test = transformer.prepare_dataset(
        df_clean, target_col='Approval Status', test_size=0.2, balance_data=True
    )
    
    # 4. Train and Compare
    trainer = ModelTrainer()
    results, best_run = trainer.train_and_compare(X_train, y_train, X_test, y_test)
    
    # 5. Save Pipeline
    trainer.save_pipeline(cleaner, transformer)
    
    # Return metrics for evaluation
    return results, best_run, X_test, y_test

if __name__ == '__main__':
    results, best_run, X_test, y_test = run_training_pipeline()
    # Regenerate evaluation plots and metrics.joblib locally
    import joblib
    meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'model_meta.joblib')
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
        from evaluation.evaluator import run_evaluation_pipeline
        run_evaluation_pipeline(meta, best_run['model'], X_test, y_test)
