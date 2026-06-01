"""
src/models/evaluate.py
Evaluation utilities — used by 06_evaluation.py and pipeline.py

Changes vs original:
  - Fixed matplotlib backend: use() must be called BEFORE pyplot import
  - Added save_evaluation_report() to write CSV + JSON for the dashboard
"""

import matplotlib
matplotlib.use("Agg")   # Must be BEFORE pyplot import

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score,
    average_precision_score, precision_recall_curve,
)


def evaluate_model(
    model,
    X_test,
    y_test,
    feature_names: list | None = None,
) -> dict:
    """
    Run full evaluation on test set.
    Returns dict with all metrics + arrays for plotting.

    Usage:
        from src.models.evaluate import evaluate_model
        results = evaluate_model(best_model, X_test, y_test)
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":      accuracy_score(y_test, y_pred),
        "precision":     precision_score(y_test, y_pred, zero_division=0),
        "recall":        recall_score(y_test, y_pred, zero_division=0),
        "f1":            f1_score(y_test, y_pred, zero_division=0),
        "auc_roc":       roc_auc_score(y_test, y_proba),
        "avg_precision": average_precision_score(y_test, y_proba),
        "y_pred":        y_pred,
        "y_proba":       y_proba,
        "report":        classification_report(
            y_test, y_pred, target_names=["Non-Buyer", "Buyer"]
        ),
    }


def plot_confusion_matrix(
    y_test,
    y_pred,
    ax=None,
    figsize=(5, 4),
) -> plt.Figure:
    """Plot a seaborn confusion matrix. Returns the figure."""
    import seaborn as sns
    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Non-Buyer", "Buyer"],
        yticklabels=["Non-Buyer", "Buyer"],
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    return fig


def plot_roc_curve(y_test, y_proba, ax=None, figsize=(5, 4)) -> plt.Figure:
    """Plot ROC curve. Returns the figure."""
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = (plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.08)
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    return fig


def build_lift_table(y_test, y_proba) -> pd.DataFrame:
    """
    Build a lift / cumulative gains table.
    Returns DataFrame with columns: pct_targeted, pct_captured, lift.
    """
    df = (
        pd.DataFrame({"y_true": y_test, "y_score": y_proba})
        .sort_values("y_score", ascending=False)
        .reset_index(drop=True)
    )
    n_total  = len(df)
    n_buyers = df["y_true"].sum()
    df["cum_buyers"]   = df["y_true"].cumsum()
    df["pct_targeted"] = (df.index + 1) / n_total
    df["pct_captured"] = df["cum_buyers"] / n_buyers
    df["lift"]         = df["pct_captured"] / df["pct_targeted"]
    return df[["pct_targeted", "pct_captured", "lift"]]


def optimal_threshold(y_test, y_proba) -> tuple[float, float]:
    """Return (threshold, f1) that maximises F1 on the given test set."""
    _, _, thresholds = precision_recall_curve(y_test, y_proba)
    f1s = [
        f1_score(y_test, (y_proba >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(f1s))
    return thresholds[best_idx], f1s[best_idx]


def save_evaluation_report(
    metrics: dict,
    lift_df: pd.DataFrame,
    feature_importance: dict,
    output_dir: Path,
) -> Path:
    """
    Save evaluation results so the dashboard can read real numbers.

    Writes:
      {output_dir}/evaluation_report.csv   — metric name / value table
      {output_dir}/dashboard_data.json     — full JSON blob for HTML dashboard

    Args:
        metrics:            Dict from evaluate_model()
        lift_df:            DataFrame from build_lift_table()
        feature_importance: {feature_name: importance_value} dict
        output_dir:         Where to save files

    Returns:
        Path to dashboard_data.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── evaluation_report.csv (used by Streamlit app.py) ────────
    scalar_metrics = {
        k: v for k, v in metrics.items()
        if isinstance(v, (int, float))
    }
    report_df = pd.DataFrame(
        [{"metric": k, "value": round(v, 4)} for k, v in scalar_metrics.items()]
    )
    report_path = output_dir / "evaluation_report.csv"
    report_df.to_csv(report_path, index=False)

    # ── dashboard_data.json (used by dashboard_index.html) ──────
    lift_rows = []
    for pct in [0.10, 0.20, 0.30, 0.40, 0.50]:
        row = lift_df[lift_df["pct_targeted"] >= pct].iloc[0]
        lift_rows.append({
            "pct":           pct,
            "pct_captured":  round(float(row["pct_captured"]), 4),
            "lift":          round(float(row["lift"]), 2),
        })

    # Lift curve for chart (sample every 50th row to keep JSON small)
    lift_curve = lift_df.iloc[::max(1, len(lift_df)//100)][
        ["pct_targeted", "pct_captured", "lift"]
    ].round(4).to_dict(orient="records")

    # ROC curve points
    y_test  = metrics.get("y_true",  None)
    y_proba = metrics["y_proba"]
    if y_test is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_points  = [
            [round(float(x), 4), round(float(y), 4)]
            for x, y in zip(fpr[::max(1, len(fpr)//100)],
                            tpr[::max(1, len(tpr)//100)])
        ]
    else:
        roc_points = []

    # Feature importance (top 10)
    top_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:10]

    dashboard_data = {
        "metrics": {
            "auc_roc":   round(scalar_metrics.get("auc_roc", 0), 4),
            "f1":        round(scalar_metrics.get("f1", 0), 4),
            "precision": round(scalar_metrics.get("precision", 0), 4),
            "recall":    round(scalar_metrics.get("recall", 0), 4),
            "accuracy":  round(scalar_metrics.get("accuracy", 0), 4),
        },
        "lift_summary":     lift_rows,
        "lift_curve":       lift_curve,
        "roc_curve":        roc_points,
        "feature_importance": {
            "labels": [f[0] for f in top_features],
            "values": [round(float(f[1]), 4) for f in top_features],
        },
    }

    json_path = output_dir / "dashboard_data.json"
    json_path.write_text(json.dumps(dashboard_data, indent=2))

    return json_path