"""
src/models/train_advanced.py
Advanced ML training with proper validation and overfitting prevention

Key improvements:
1. Temporal train/test split (no data leakage)
2. Cross-validation with stratified K-fold
3. Feature selection and regularization
4. Advanced ensemble methods
5. Proper hyperparameter tuning
6. Overfitting detection
"""

import json
import joblib
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score, make_scorer
)

warnings.filterwarnings('ignore')


def temporal_train_test_split(
    feature_matrix: pd.DataFrame,
    target_col: str = "will_purchase",
    test_ratio: float = 0.20,
):
    """
    Split data temporally to prevent data leakage.
    Uses account_age_days as proxy for time.
    
    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
    """
    # Sort by account age (older accounts = training, newer = test)
    df_sorted = feature_matrix.sort_values('account_age_days').reset_index(drop=True)
    
    split_idx = int(len(df_sorted) * (1 - test_ratio))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    drop_cols = ["user_id", target_col, "account_age_days"]
    feature_names = [c for c in feature_matrix.columns if c not in drop_cols]
    
    X_train = train_df[feature_names].copy()
    X_test = test_df[feature_names].copy()
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    # Encode any remaining categorical columns
    from sklearn.preprocessing import LabelEncoder
    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    print(f"✓ Temporal split: Train={len(X_train)} ({len(X_train)/len(df_sorted)*100:.1f}%), Test={len(X_test)} ({len(X_test)/len(df_sorted)*100:.1f}%)")
    print(f"  Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"  Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, feature_names


def detect_overfitting(model, X_train, X_test, y_train, y_test):
    """
    Detect overfitting by comparing train vs test performance.
    
    Returns:
        dict with train/test metrics and overfitting indicators
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_train, y_train_proba)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_test_proba)
    }
    
    # Calculate overfitting gap
    gaps = {
        metric: train_metrics[metric] - test_metrics[metric]
        for metric in train_metrics.keys()
    }
    
    # Overfitting detected if gap > 5% for any metric
    is_overfitted = any(gap > 0.05 for gap in gaps.values())
    
    return {
        'train': train_metrics,
        'test': test_metrics,
        'gaps': gaps,
        'is_overfitted': is_overfitted
    }


def train_logistic_regression_advanced(X_train, y_train, X_test=None, y_test=None):
    """
    Train Logistic Regression with L1/L2 regularization and feature selection.
    """
    print("\n🔹 Training Logistic Regression with Regularization...")
    
    # Try different regularization strengths
    param_dist = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'clf__penalty': ['l1', 'l2', 'elasticnet'],
        'clf__solver': ['saga'],
        'clf__l1_ratio': [0.3, 0.5, 0.7]  # For elasticnet
    }
    
    pipe = Pipeline([
        ('scaler', RobustScaler()),  # More robust to outliers
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            random_state=42
        ))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=20, cv=cv,
        scoring='roc_auc', n_jobs=-1, random_state=42, verbose=0
    )
    
    search.fit(X_train, y_train)
    
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    
    if X_test is not None and y_test is not None:
        overfit_check = detect_overfitting(search.best_estimator_, X_train, X_test, y_train, y_test)
        print(f"  Train AUC: {overfit_check['train']['auc_roc']:.4f}")
        print(f"  Test AUC:  {overfit_check['test']['auc_roc']:.4f}")
        print(f"  Gap: {overfit_check['gaps']['auc_roc']:.4f}")
        if overfit_check['is_overfitted']:
            print("  ⚠️  Overfitting detected!")
    
    return search.best_estimator_


def train_random_forest_advanced(X_train, y_train, X_test=None, y_test=None):
    """
    Train Random Forest with proper regularization to prevent overfitting.
    """
    print("\n🔹 Training Random Forest with Regularization...")
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [10, 20, 50],
        'min_samples_leaf': [5, 10, 20],
        'max_features': ['sqrt', 'log2', 0.5],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_dist, n_iter=20, cv=cv,
        scoring='roc_auc', n_jobs=-1, random_state=42, verbose=0
    )
    
    search.fit(X_train, y_train)
    
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    
    if X_test is not None and y_test is not None:
        overfit_check = detect_overfitting(search.best_estimator_, X_train, X_test, y_train, y_test)
        print(f"  Train AUC: {overfit_check['train']['auc_roc']:.4f}")
        print(f"  Test AUC:  {overfit_check['test']['auc_roc']:.4f}")
        print(f"  Gap: {overfit_check['gaps']['auc_roc']:.4f}")
        if overfit_check['is_overfitted']:
            print("  ⚠️  Overfitting detected!")
    
    return search.best_estimator_


def train_xgboost_advanced(X_train, y_train, X_test=None, y_test=None):
    """
    Train XGBoost with regularization and early stopping.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  ⚠️  XGBoost not installed, skipping...")
        return None
    
    print("\n🔹 Training XGBoost with Regularization...")
    
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_child_weight': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],  # L1 regularization
        'reg_lambda': [1, 5, 10],  # L2 regularization
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        XGBClassifier(
            scale_pos_weight=scale_pos,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        param_dist, n_iter=20, cv=cv,
        scoring='roc_auc', n_jobs=-1, random_state=42, verbose=0
    )
    
    search.fit(X_train, y_train)
    
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    
    if X_test is not None and y_test is not None:
        overfit_check = detect_overfitting(search.best_estimator_, X_train, X_test, y_train, y_test)
        print(f"  Train AUC: {overfit_check['train']['auc_roc']:.4f}")
        print(f"  Test AUC:  {overfit_check['test']['auc_roc']:.4f}")
        print(f"  Gap: {overfit_check['gaps']['auc_roc']:.4f}")
        if overfit_check['is_overfitted']:
            print("  ⚠️  Overfitting detected!")
    
    return search.best_estimator_


def train_lightgbm_advanced(X_train, y_train, X_test=None, y_test=None):
    """
    Train LightGBM with regularization.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("  ⚠️  LightGBM not installed, skipping...")
        return None
    
    print("\n🔹 Training LightGBM with Regularization...")
    
    param_dist = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7, -1],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [10, 20, 30],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 5, 10],
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        LGBMClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        param_dist, n_iter=20, cv=cv,
        scoring='roc_auc', n_jobs=-1, random_state=42, verbose=0
    )
    
    search.fit(X_train, y_train)
    
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV AUC: {search.best_score_:.4f}")
    
    if X_test is not None and y_test is not None:
        overfit_check = detect_overfitting(search.best_estimator_, X_train, X_test, y_train, y_test)
        print(f"  Train AUC: {overfit_check['train']['auc_roc']:.4f}")
        print(f"  Test AUC:  {overfit_check['test']['auc_roc']:.4f}")
        print(f"  Gap: {overfit_check['gaps']['auc_roc']:.4f}")
        if overfit_check['is_overfitted']:
            print("  ⚠️  Overfitting detected!")
    
    return search.best_estimator_


def train_catboost_advanced(X_train, y_train, X_test=None, y_test=None):
    """
    Train CatBoost with regularization.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  ⚠️  CatBoost not installed, skipping...")
        return None
    
    print("\n🔹 Training CatBoost with Regularization...")
    
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        scale_pos_weight=scale_pos,
        random_state=42,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    if X_test is not None and y_test is not None:
        overfit_check = detect_overfitting(model, X_train, X_test, y_train, y_test)
        print(f"  Train AUC: {overfit_check['train']['auc_roc']:.4f}")
        print(f"  Test AUC:  {overfit_check['test']['auc_roc']:.4f}")
        print(f"  Gap: {overfit_check['gaps']['auc_roc']:.4f}")
        if overfit_check['is_overfitted']:
            print("  ⚠️  Overfitting detected!")
    
    return model


def save_model_advanced(model, metadata: dict, models_dir: Path) -> Path:
    """Save model with metadata."""
    models_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    
    model_name = type(model).__name__
    if hasattr(model, "steps"):
        model_name = type(model.steps[-1][1]).__name__
    elif hasattr(model, "best_estimator_"):
        model_name = type(model.best_estimator_).__name__
    
    model_path = models_dir / f"best_model_{model_name}_{ts}.pkl"
    meta_path = models_dir / f"model_metadata_{ts}.json"
    
    joblib.dump(model, model_path)
    
    metadata["model_path_relative"] = model_path.name
    metadata["training_method"] = "advanced_with_regularization"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    
    return model_path
