"""
Phase 6 — Evaluation & Interpretation
========================================
Full evaluation of the best model:
  - Metrics (AUC-ROC, F1, Precision, Recall, Accuracy)
  - Confusion matrix & ROC curve
  - SHAP global & local explanations
  - Lift / cumulative gains curve
  - Optimal threshold analysis
  - Saves dashboard_data.json for the HTML dashboard (REAL numbers)
  - Saves evaluation_report.csv for the Streamlit app

Run: python notebooks/06_evaluation.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import warnings
warnings.filterwarnings("ignore")

import json
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from config.settings import settings
from src.models.train import prepare_data
from src.models.evaluate import (
    evaluate_model,
    build_lift_table,
    optimal_threshold,
    plot_confusion_matrix,
    plot_roc_curve,
    save_evaluation_report,
)

FIGS = settings.FIGURES_DIR


def load_model_and_meta():
    """Load the latest saved model and its metadata."""
    model_dir = settings.MODEL_DIR
    pkl_files  = sorted(model_dir.glob("best_model_*.pkl"))
    meta_files = sorted(model_dir.glob("model_metadata_*.json"))

    if not pkl_files:
        return None, {}

    model = joblib.load(pkl_files[-1])
    meta  = json.loads(meta_files[-1].read_text()) if meta_files else {}
    return model, meta


def main():
    """Main function to run model evaluation."""

    print("=" * 80)
    print("PHASE 6: EVALUATION & INTERPRETATION (will_purchase)")
    print("=" * 80)

    # ── Load model ───────────────────────────────────────────────
    print("\n1. Loading trained model ...")
    model, meta = load_model_and_meta()
    if model is None:
        print("❌  No trained model found — run Phase 5 first.")
        return None

    print(f"   Model type  : {meta.get('model_type', type(model).__name__)}")
    print(f"   AUC-ROC     : {meta.get('test_auc', 'N/A')}")
    print(f"   F1 Score    : {meta.get('test_f1',  'N/A')}")

    # ── Load feature matrix ──────────────────────────────────────
    print("\n2. Loading feature matrix ...")
    fm_path = settings.PROCESSED_DATA_DIR / "feature_matrix.parquet"
    if not fm_path.exists():
        fm_path = settings.PROCESSED_DATA_DIR / "feature_matrix.csv"
    if not fm_path.exists():
        print("❌  feature_matrix not found — run Phase 4 first.")
        return None

    fm = pd.read_parquet(fm_path) if fm_path.suffix == ".parquet" else pd.read_csv(fm_path)

    # ── Reproduce exact same split used in training ──────────────
    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        fm, target_col="will_purchase", test_size=0.20, random_state=42
    )
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Full metrics ─────────────────────────────────────────────
    print("\n3. Performance Metrics:")
    results = evaluate_model(model, X_test, y_test)
    print(results["report"])

    scalar = {k: v for k, v in results.items() if isinstance(v, (int, float))}
    print("   Metric summary:")
    for k, v in scalar.items():
        print(f"   {k:<20} {v:.4f}")

    # ── Confusion matrix + ROC ───────────────────────────────────
    print("\n4. Plotting confusion matrix and ROC curve ...")
    FIGS.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 6 — Model Evaluation", fontsize=13, fontweight="bold")
    plot_confusion_matrix(y_test, y_pred, ax=axes[0])
    plot_roc_curve(y_test, y_proba, ax=axes[1])
    plt.tight_layout()
    p = FIGS / "06_roc_pr_threshold.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {p}")

    # ── Lift curve ───────────────────────────────────────────────
    print("\n5. Building lift / cumulative gains table ...")
    lift_df = build_lift_table(y_test.values, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Lift & Cumulative Gains", fontsize=13, fontweight="bold")

    axes[0].plot(lift_df["pct_targeted"] * 100, lift_df["pct_captured"] * 100,
                 color="#4f8ef7", lw=2, label="Model")
    axes[0].plot([0, 100], [0, 100], "k--", lw=1, label="Random")
    axes[0].set_title("Cumulative Gains Curve")
    axes[0].set_xlabel("% Users Targeted"); axes[0].set_ylabel("% Buyers Captured")
    axes[0].legend()

    axes[1].plot(lift_df["pct_targeted"] * 100, lift_df["lift"],
                 color="#fc8d62", lw=2)
    axes[1].axhline(1, color="k", linestyle="--", lw=1)
    axes[1].set_title("Lift Curve")
    axes[1].set_xlabel("% Users Targeted"); axes[1].set_ylabel("Lift")

    plt.tight_layout()
    p = FIGS / "06_lift_curve.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {p}")

    # Summary table
    print("\n   Lift Summary:")
    print(f"   {'Top %':<10} {'% Buyers Captured':<22} {'Lift':>6}")
    for pct in [0.10, 0.20, 0.30, 0.50]:
        row = lift_df[lift_df["pct_targeted"] >= pct].iloc[0]
        print(f"   {pct*100:.0f}%{'':7} {row['pct_captured']*100:.1f}%{'':14} "
              f"{row['lift']:.2f}×")

    # ── Optimal threshold ────────────────────────────────────────
    print("\n6. Optimal threshold analysis ...")
    thresh, best_f1 = optimal_threshold(y_test.values, y_proba)
    print(f"   Optimal threshold : {thresh:.4f}")
    print(f"   F1 at threshold   : {best_f1:.4f}")
    print(f"   Default (0.5) F1  : {results['f1']:.4f}")

    # ── SHAP explanations ────────────────────────────────────────
    try:
        import shap
        print("\n7. Computing SHAP values ...")

        # Use underlying estimator if Pipeline
        estimator = model
        X_shap    = X_test
        if hasattr(model, "steps"):
            # Pipeline: transform data through all but last step
            from sklearn.pipeline import Pipeline as _Pipeline
            for name_, step in model.steps[:-1]:
                X_shap = step.transform(X_shap)
            estimator = model.steps[-1][1]

        explainer  = shap.TreeExplainer(estimator) if hasattr(estimator, "feature_importances_") \
                     else shap.LinearExplainer(estimator, X_shap)
        shap_values = explainer.shap_values(X_shap)

        # For binary classifiers shap_values is sometimes [neg_class, pos_class]
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Global summary
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv, X_shap, feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.title("SHAP Feature Importance (Global)")
        plt.tight_layout()
        p = FIGS / "06_shap_bar.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Saved → {p}")

        # SHAP summary dot plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_shap, feature_names=feature_names, show=False)
        plt.title("SHAP Summary Plot")
        plt.tight_layout()
        p = FIGS / "06_shap_summary.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Saved → {p}")

        # Extract SHAP feature importance for dashboard
        shap_importance = dict(zip(feature_names, np.abs(sv).mean(axis=0)))

    except ImportError:
        print("   ⚠️  shap not installed — using model feature_importances_ instead")
        shap_importance = {}
        if hasattr(estimator if "estimator" in dir() else model, "feature_importances_"):
            est = model if not hasattr(model, "steps") else model.steps[-1][1]
            shap_importance = dict(zip(feature_names, est.feature_importances_))
    except Exception as e:
        print(f"   ⚠️  SHAP failed: {e}")
        shap_importance = {}

    # ── Feature importance (fallback) ────────────────────────────
    feature_importance = shap_importance
    if not feature_importance:
        est = model if not hasattr(model, "steps") else model.steps[-1][1]
        if hasattr(est, "feature_importances_"):
            feature_importance = dict(zip(feature_names, est.feature_importances_))

    # ── Save dashboard data (REAL numbers!) ──────────────────────
    print("\n8. Saving dashboard data (real numbers, no hardcoding) ...")
    results["y_true"] = y_test.values  # needed for ROC points in save_evaluation_report

    json_path = save_evaluation_report(
        metrics=results,
        lift_df=lift_df,
        feature_importance=feature_importance,
        output_dir=settings.OUTPUT_DIR,
    )
    print(f"   Saved evaluation_report.csv → {settings.OUTPUT_DIR / 'evaluation_report.csv'}")
    print(f"   Saved dashboard_data.json   → {json_path}")

    print("\n" + "=" * 80)
    print("PHASE 6 COMPLETE — Dashboard data ready!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
