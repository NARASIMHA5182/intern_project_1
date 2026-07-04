import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib
from utils.logger import logger

class ModelEvaluator:
    """
    Evaluates trained models and generates evaluation charts.
    """
    def __init__(self, image_dir=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.image_dir = image_dir or os.path.join(base_dir, 'static', 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        
    def evaluate_model(self, model, X_test, y_test, feature_names=None):
        """
        Calculates accuracy, precision, recall, specificity, sensitivity, 
        confusion matrix, and saves metrics visualization.
        """
        logger.info("Evaluating model performance...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Core Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Specificity & Sensitivity
        sensitivity = recall  # Sensitivity is equal to recall (True Positive Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'roc_auc': float(roc_auc),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'classification_report': report
        }
        
        # Generate Plots
        self.plot_confusion_matrix(cm)
        self.plot_roc_curve(fpr, tpr, roc_auc)
        if feature_names is not None:
            self.plot_feature_importance(model, feature_names)
            
        logger.info(f"Model evaluation complete. AUC: {roc_auc:.4f} | Accuracy: {accuracy:.4f}")
        return metrics
        
    def plot_confusion_matrix(self, cm):
        """
        Generates and saves Confusion Matrix.
        """
        plt.figure(figsize=(6, 5))
        # Blue gradient styling
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Rejected', 'Approved'],
                    yticklabels=['Rejected', 'Approved'],
                    annot_kws={"size": 14, "weight": "bold"})
        plt.title('Confusion Matrix', fontsize=14, pad=15)
        plt.ylabel('Actual Status', fontsize=12)
        plt.xlabel('Predicted Status', fontsize=12)
        plt.tight_layout()
        
        path = os.path.join(self.image_dir, 'confusion_matrix.png')
        plt.savefig(path, dpi=300, transparent=True)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {path}")
        
    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """
        Generates and saves ROC Curve.
        """
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='#0052cc', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='#e1e1e1', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=15)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        path = os.path.join(self.image_dir, 'roc_curve.png')
        plt.savefig(path, dpi=300, transparent=True)
        plt.close()
        logger.info(f"Saved ROC curve plot to {path}")
        
    def plot_feature_importance(self, model, feature_names):
        """
        Generates and saves Feature Importance plot.
        """
        importance = None
        
        # Extract feature importance based on model type
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            
        if importance is None:
            logger.warning("Could not extract feature importance. Skipping plot.")
            return
            
        # Select top 15 features for readability
        features_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False).head(15)
        
        plt.figure(figsize=(8, 6))
        # Blue gradient styling
        sns.barplot(x='Importance', y='Feature', data=features_df, palette='Blues_r')
        plt.title('Top 15 Most Important Features', fontsize=14, pad=15)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        path = os.path.join(self.image_dir, 'feature_importance.png')
        plt.savefig(path, dpi=300, transparent=True)
        plt.close()
        logger.info(f"Saved feature importance plot to {path}")

def run_evaluation_pipeline(model_meta, best_model, X_test, y_test):
    """
    Invokes the evaluator using test sets. Saves metrics.
    """
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_model(
        best_model, 
        X_test, 
        y_test, 
        feature_names=model_meta.get('features')
    )
    
    # Save metrics evaluation file alongside models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, 'models', 'metrics.joblib')
    joblib.dump(metrics, metrics_path)
    logger.info("Evaluation metrics serialized.")
    return metrics
