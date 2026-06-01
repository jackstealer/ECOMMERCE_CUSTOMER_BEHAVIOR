"""
Phase 5 — Model Development
==============================
Trains multiple classifiers on the feature matrix to predict will_purchase.

Models trained:
  1. Logistic Regression (baseline, inside Pipeline with StandardScaler)
  2. Random Forest
  3. XGBoost (if installed)
  4. LightGBM (if installed)

Best model (by AUC-ROC) is saved to data/outputs/models/

Run: python notebooks/05_modeling.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from config.settings import settings
from src.models.train import (
    prepare_data,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    save_model,
)
from src.models.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve

FIGS = settings.FIGURES_DIR


def main():
    """Main function to run model development pipeline."""

    print("=" * 80)
    print("PHASE 5: MODEL DEVELOPMENT (target: will_purchase)")
    print("=" * 80)

    # ── Load feature matrix ──────────────────────────────────────
    print("\n1. Loading feature matrix ...")
    fm_path = settings.PROCESSED_DATA_DIR / "feature_matrix.parquet"
    if not fm_path.exists():
        fm_path = settings.PROCESSED_DATA_DIR / "feature_matrix.csv"
    if not fm_path.exists():
        print("❌  feature_matrix not found — run Phase 4 first.")
        return None, None

    fm = pd.read_parquet(fm_path) if fm_path.suffix == ".parquet" else pd.read_csv(fm_path)
    print(f"   Shape: {fm.shape}")
    print(f"   Columns: {list(fm.columns)}")

    # ── Prepare train/test split ─────────────────────────────────
    print("\n2. Preparing data for modeling (target: will_purchase) ...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        fm, target_col="will_purchase", test_size=0.20, random_state=42
    )
    print(f"   Training set : {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"   Test set     : {X_test.shape[0]:,} rows")
    print(f"   Buyers in train : {y_train.sum():,}  ({y_train.mean()*100:.1f}%)")
    print(f"   Buyers in test  : {y_test.sum():,}   ({y_test.mean()*100:.1f}%)")

    # ── Train all models ─────────────────────────────────────────
    print("\n3. Training models ...")
    models   = {}
    results  = {}

    # Logistic Regression
    print("   a) Logistic Regression ...")
    lr = train_logistic_regression(X_train, y_train)
    models["Logistic Regression"] = lr
    results["Logistic Regression"] = _score(lr, X_test, y_test)

    # Random Forest
    print("   b) Random Forest ...")
    rf = train_random_forest(X_train, y_train)
    models["Random Forest"] = rf
    results["Random Forest"] = _score(rf, X_test, y_test)

    # XGBoost
    print("   c) XGBoost ...")
    xgb = train_xgboost(X_train, y_train)
    if xgb:
        models["XGBoost"] = xgb
        results["XGBoost"] = _score(xgb, X_test, y_test)

    # LightGBM
    print("   d) LightGBM ...")
    lgbm = train_lightgbm(X_train, y_train)
    if lgbm:
        models["LightGBM"] = lgbm
        results["LightGBM"] = _score(lgbm, X_test, y_test)

    # ── Model comparison table ───────────────────────────────────
    print("\n4. Model Comparison:")
    print("=" * 85)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>9}"
          f" {'F1':>9} {'AUC-ROC':>9}")
    print("=" * 85)
    for name, m in results.items():
        print(f"  {name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>9.4f} {m['f1']:>9.4f} {m['roc_auc']:>9.4f}")
    print("=" * 85)

    # ── Select best model ────────────────────────────────────────
    best_name  = max(results, key=lambda x: results[x]["roc_auc"])
    best_model = models[best_name]
    best_m     = results[best_name]
    print(f"\n5. Best model: {best_name}")
    print(f"   AUC-ROC  : {best_m['roc_auc']:.4f}")
    print(f"   F1 Score : {best_m['f1']:.4f}")

    # ── Feature importance ───────────────────────────────────────
    feature_importance = {}
    if hasattr(best_model, "feature_importances_"):
        fi = dict(zip(feature_names, best_model.feature_importances_))
        feature_importance = fi
        top_10 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n6. Top 10 Features ({best_name}):")
        for feat, imp in top_10:
            print(f"   {feat:<40}  {imp:.4f}")
    elif hasattr(best_model, "steps"):
        # Pipeline (Logistic Regression case)
        clf = best_model.steps[-1][1]
        if hasattr(clf, "coef_"):
            fi = dict(zip(feature_names, np.abs(clf.coef_[0])))
            feature_importance = fi

    # ── Visualisation ────────────────────────────────────────────
    print("\n7. Saving evaluation plots ...")
    FIGS.mkdir(parents=True, exist_ok=True)

    # Confusion matrix + ROC side-by-side
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{best_name} — Evaluation", fontsize=13, fontweight="bold")
    plot_confusion_matrix(y_test, y_pred, ax=axes[0])
    plot_roc_curve(y_test, y_proba, ax=axes[1])
    plt.tight_layout()
    path = FIGS / "05_model_evaluation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {path}")

    # Feature importance bar chart
    if feature_importance:
        top = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        fig, ax = plt.subplots(figsize=(10, 6))
        feats, imps = zip(*top)
        ax.barh(list(reversed(feats)), list(reversed(imps)), color="#4f8ef7", alpha=0.85)
        ax.set_title(f"Feature Importances — {best_name}")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        path = FIGS / "05_feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"   Saved → {path}")

    # ── Save best model ──────────────────────────────────────────
    print(f"\n8. Saving best model ({best_name}) ...")
    metadata = {
        "model_type":    best_name,
        "target_col":    "will_purchase",
        "n_features":    len(feature_names),
        "feature_names": feature_names,
        "test_auc":      round(best_m["roc_auc"], 4),
        "test_f1":       round(best_m["f1"], 4),
        "test_precision":round(best_m["precision"], 4),
        "test_recall":   round(best_m["recall"], 4),
        "test_accuracy": round(best_m["accuracy"], 4),
        "best_params":   getattr(best_model, "best_params_", {}),
    }
    model_path = save_model(best_model, metadata, settings.MODEL_DIR)
    print(f"   Saved → {model_path}")

    print("\n" + "=" * 80)
    print("PHASE 5 COMPLETE")
    print("=" * 80)

    return models, results, X_test, y_test, y_proba, feature_importance, feature_names


def _score(model, X_test, y_test) -> dict:
    """Compute classification metrics for a model."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall":    recall_score(y_test, y_pred, zero_division=0),
        "f1":        f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }


if __name__ == "__main__":
    main()
