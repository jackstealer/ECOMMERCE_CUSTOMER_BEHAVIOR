"""
Phase 5 — Model Development

This script covers:
- Train/test split
- Model selection and training
- Hyperparameter tuning
- Model comparison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
sys.path.append('../src')
from models.train import train_random_forest, train_xgboost, prepare_data
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function to run model development pipeline."""
    
    print("=" * 80)
    print("PHASE 5: MODEL DEVELOPMENT")
    print("=" * 80)
    
    # Load feature-engineered data
    print("\n1. Loading feature-engineered data...")
    df = pd.read_csv('../data/processed/customer_behavior_features.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Prepare data for modeling
    print("\n2. Preparing data for modeling...")
    X_train, X_test, y_train, y_test = prepare_data(df, target_col='churned')
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Target distribution in training set:")
    print(f"  Not Churned: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
    print(f"  Churned: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
    
    # Dictionary to store models and their performance
    models = {}
    results = {}
    
    # Model 1: Logistic Regression
    print("\n3. Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1_score': f1_score(y_test, y_pred_lr),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_lr)
    }
    print("Logistic Regression trained successfully!")
    
    # Model 2: Random Forest
    print("\n4. Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    
    y_pred_rf = rf_model.predict(X_test)
    y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1_score': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_rf)
    }
    print("Random Forest trained successfully!")
    
    # Model 3: Gradient Boosting
    print("\n5. Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_model
    
    y_pred_gb = gb_model.predict(X_test)
    y_pred_proba_gb = gb_model.predict_proba(X_test)[:, 1]
    
    results['Gradient Boosting'] = {
        'accuracy': accuracy_score(y_test, y_pred_gb),
        'precision': precision_score(y_test, y_pred_gb),
        'recall': recall_score(y_test, y_pred_gb),
        'f1_score': f1_score(y_test, y_pred_gb),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_gb)
    }
    print("Gradient Boosting trained successfully!")
    
    # Model 4: XGBoost (if available)
    try:
        print("\n6. Training XGBoost...")
        xgb_model = train_xgboost(X_train, y_train)
        models['XGBoost'] = xgb_model
        
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        
        results['XGBoost'] = {
            'accuracy': accuracy_score(y_test, y_pred_xgb),
            'precision': precision_score(y_test, y_pred_xgb),
            'recall': recall_score(y_test, y_pred_xgb),
            'f1_score': f1_score(y_test, y_pred_xgb),
            'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb)
        }
        print("XGBoost trained successfully!")
    except Exception as e:
        print(f"XGBoost training skipped: {e}")
    
    # Model comparison
    print("\n7. Model Performance Comparison:")
    print("=" * 100)
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("=" * 100)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<20} "
              f"{metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} "
              f"{metrics['roc_auc']:<12.4f}")
    
    print("=" * 100)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = models[best_model_name]
    
    print(f"\n8. Best Model: {best_model_name}")
    print(f"   ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}")
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n9. Top 10 Most Important Features ({best_model_name}):")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
    
    # Save predictions
    print("\n10. Saving predictions...")
    predictions_df = pd.DataFrame({
        'actual': y_test,
        'predicted': best_model.predict(X_test),
        'probability': best_model.predict_proba(X_test)[:, 1]
    })
    predictions_df.to_csv('../data/outputs/predictions.csv', index=False)
    print("Predictions saved to: ../data/outputs/predictions.csv")
    
    print("\n" + "=" * 80)
    print("PHASE 5 COMPLETE: Model Development Finished!")
    print("=" * 80)
    
    return models, results, X_test, y_test


if __name__ == "__main__":
    models, results, X_test, y_test = main()
