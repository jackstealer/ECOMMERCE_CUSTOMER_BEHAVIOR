"""
Model training scripts for customer behavior prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib


def prepare_data(df, target_col='churned', test_size=0.2, random_state=42):
    """
    Prepare data for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Feature-engineered dataframe
    target_col : str
        Name of target column
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    # Select features (exclude ID and target)
    feature_cols = [col for col in df.columns 
                   if col not in ['customer_id', target_col, 'gender']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, 
                          random_state=random_state, stratify=y)


def train_random_forest(X_train, y_train, params=None):
    """
    Train Random Forest classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    params : dict, optional
        Model parameters
        
    Returns:
    --------
    RandomForestClassifier
        Trained model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def train_xgboost(X_train, y_train, params=None):
    """
    Train XGBoost classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    params : dict, optional
        Model parameters
        
    Returns:
    --------
    XGBClassifier
        Trained model
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
    
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    filepath : str
        Path to save model
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to model file
        
    Returns:
    --------
    sklearn model
        Loaded model
    """
    return joblib.load(filepath)
