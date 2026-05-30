"""
Model evaluation utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics


def plot_confusion_matrix(y_test, y_pred, save_path=None):
    """
    Plot confusion matrix.
    
    Parameters:
    -----------
    y_test : pd.Series
        True labels
    y_pred : np.array
        Predicted labels
    save_path : str, optional
        Path to save figure
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    save_path : str, optional
        Path to save figure
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances')
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, save_path=None):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_test : pd.Series
        True labels
    y_pred_proba : np.array
        Predicted probabilities
    save_path : str, optional
        Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
