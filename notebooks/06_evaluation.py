"""
Phase 6 — Evaluation & Interpretation

This script covers:
- Model performance metrics
- Confusion matrix analysis
- Feature importance
- Model interpretation (SHAP values)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, roc_auc_score, precision_recall_curve
)
import sys
sys.path.append('../src')
from models.evaluate import evaluate_model, plot_confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main function to run model evaluation."""
    
    print("=" * 80)
    print("PHASE 6: MODEL EVALUATION & INTERPRETATION")
    print("=" * 80)
    
    # Load predictions
    print("\n1. Loading predictions...")
    try:
        predictions_df = pd.read_csv('../data/outputs/predictions.csv')
        print(f"Predictions loaded: {predictions_df.shape[0]} samples")
    except FileNotFoundError:
        print("Error: predictions.csv not found. Please run Phase 5 (modeling) first.")
        return
    
    y_test = predictions_df['actual']
    y_pred = predictions_df['predicted']
    y_pred_proba = predictions_df['probability']
    
    # Classification Report
    print("\n2. Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Not Churned', 'Churned']))
    
    # Confusion Matrix
    print("\n3. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('../reports/figures/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Saved confusion matrix to: ../reports/figures/confusion_matrix.png")
    plt.show()
    
    # ROC Curve
    print("\n4. ROC Curve Analysis:")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../reports/figures/roc_curve.png', dpi=300, bbox_inches='tight')
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Saved ROC curve to: ../reports/figures/roc_curve.png")
    plt.show()
    
    # Precision-Recall Curve
    print("\n5. Precision-Recall Curve:")
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../reports/figures/precision_recall_curve.png', 
                dpi=300, bbox_inches='tight')
    print("Saved precision-recall curve to: ../reports/figures/precision_recall_curve.png")
    plt.show()
    
    # Prediction distribution
    print("\n6. Prediction Probability Distribution:")
    plt.figure(figsize=(10, 6))
    
    # Separate probabilities by actual class
    churned_probs = y_pred_proba[y_test == 1]
    not_churned_probs = y_pred_proba[y_test == 0]
    
    plt.hist(not_churned_probs, bins=50, alpha=0.7, label='Not Churned (Actual)', 
             color='green', edgecolor='black')
    plt.hist(churned_probs, bins=50, alpha=0.7, label='Churned (Actual)', 
             color='red', edgecolor='black')
    plt.xlabel('Predicted Probability of Churn', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../reports/figures/probability_distribution.png', 
                dpi=300, bbox_inches='tight')
    print("Saved probability distribution to: ../reports/figures/probability_distribution.png")
    plt.show()
    
    # Model performance metrics
    print("\n7. Detailed Performance Metrics:")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc
    }
    
    print("\nPerformance Metrics:")
    print("=" * 40)
    for metric, value in metrics.items():
        print(f"{metric:<15}: {value:.4f} ({value*100:.2f}%)")
    print("=" * 40)
    
    # Error analysis
    print("\n8. Error Analysis:")
    false_positives = ((y_pred == 1) & (y_test == 0)).sum()
    false_negatives = ((y_pred == 0) & (y_test == 1)).sum()
    true_positives = ((y_pred == 1) & (y_test == 1)).sum()
    true_negatives = ((y_pred == 0) & (y_test == 0)).sum()
    
    print(f"True Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    
    # Business impact analysis
    print("\n9. Business Impact Analysis:")
    total_customers = len(y_test)
    actual_churned = y_test.sum()
    predicted_churned = y_pred.sum()
    correctly_identified = true_positives
    
    print(f"Total customers evaluated: {total_customers}")
    print(f"Actual churned customers: {actual_churned} ({actual_churned/total_customers*100:.1f}%)")
    print(f"Predicted churned customers: {predicted_churned} ({predicted_churned/total_customers*100:.1f}%)")
    print(f"Correctly identified churners: {correctly_identified} ({correctly_identified/actual_churned*100:.1f}% of actual)")
    print(f"Missed churners: {false_negatives} ({false_negatives/actual_churned*100:.1f}% of actual)")
    
    # Threshold analysis
    print("\n10. Optimal Threshold Analysis:")
    from sklearn.metrics import f1_score
    
    thresholds_to_test = np.arange(0.3, 0.8, 0.05)
    threshold_results = []
    
    for threshold in thresholds_to_test:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_threshold)
        precision = precision_score(y_test, y_pred_threshold)
        recall = recall_score(y_test, y_pred_threshold)
        threshold_results.append({
            'threshold': threshold,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        })
    
    threshold_df = pd.DataFrame(threshold_results)
    best_threshold = threshold_df.loc[threshold_df['f1_score'].idxmax()]
    
    print(f"\nBest threshold: {best_threshold['threshold']:.2f}")
    print(f"F1-Score at best threshold: {best_threshold['f1_score']:.4f}")
    print(f"Precision at best threshold: {best_threshold['precision']:.4f}")
    print(f"Recall at best threshold: {best_threshold['recall']:.4f}")
    
    # Plot threshold analysis
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['threshold'], threshold_df['f1_score'], 
             marker='o', label='F1-Score', linewidth=2)
    plt.plot(threshold_df['threshold'], threshold_df['precision'], 
             marker='s', label='Precision', linewidth=2)
    plt.plot(threshold_df['threshold'], threshold_df['recall'], 
             marker='^', label='Recall', linewidth=2)
    plt.axvline(x=best_threshold['threshold'], color='red', 
                linestyle='--', label=f'Best Threshold ({best_threshold["threshold"]:.2f})')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Threshold Analysis', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../reports/figures/threshold_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("\nSaved threshold analysis to: ../reports/figures/threshold_analysis.png")
    plt.show()
    
    print("\n" + "=" * 80)
    print("PHASE 6 COMPLETE: Model Evaluation Finished!")
    print("=" * 80)
    
    return metrics, threshold_df


if __name__ == "__main__":
    metrics, threshold_df = main()
