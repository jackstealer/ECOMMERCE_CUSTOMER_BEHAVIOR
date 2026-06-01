"""
src/models/train.py
Model training utilities — used by 05_modeling.py and pipeline.py

Changes vs original:
  - Added prepare_data() function (was imported but didn't exist)
  - train_logistic_regression returns Pipeline (scaler + model) for portability
  - save_model stores relative path to registry file
"""

import json
import joblib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.pipeline          import Pipeline
from sklearn.utils.class_weight import compute_class_weight


# ── Data preparation ─────────────────────────────────────────────

def prepare_data(
    feature_matrix: pd.DataFrame,
    target_col: str = "will_purchase",
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Split feature matrix into train/test sets.

    Args:
        feature_matrix: Full feature DataFrame (including target column)
        target_col:     Name of the target column
        test_size:      Fraction of data for test set
        random_state:   Reproducibility seed

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)
    """
    if target_col not in feature_matrix.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(feature_matrix.columns)}"
        )

    drop_cols = ["user_id", target_col]
    feature_names = [c for c in feature_matrix.columns if c not in drop_cols]

    X = feature_matrix[feature_names].copy()

    # Label-encode any remaining object/string columns so all models receive numeric data
    from sklearn.preprocessing import LabelEncoder
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    feature_names = list(X.columns)
    y = feature_matrix[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_names


# ── Class weight helper ──────────────────────────────────────────

def get_class_weights(y: pd.Series) -> dict:
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), weights.tolist()))


# ── Model trainers ───────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, **kwargs):
    """Fit a balanced Logistic Regression inside a Pipeline and return it."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42, **kwargs
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train, **kwargs):
    """Fit a balanced Random Forest and return the fitted model."""
    params = dict(
        n_estimators=200, max_depth=10, class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, **kwargs):
    """Fit XGBoost with automatic scale_pos_weight. Returns None if not installed."""
    try:
        from xgboost import XGBClassifier
        scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        params = dict(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            scale_pos_weight=scale_pos, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        params.update(kwargs)
        params.pop("use_label_encoder", None)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model
    except ImportError:
        print("XGBoost not installed — skipping.")
        return None


def train_lightgbm(X_train, y_train, **kwargs):
    """Fit LightGBM. Returns None if not installed."""
    try:
        from lightgbm import LGBMClassifier
        params = dict(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=40, class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        params.update(kwargs)
        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        return model
    except ImportError:
        print("LightGBM not installed — skipping.")
        return None


# ── Hyperparameter tuning ────────────────────────────────────────

def tune_model(
    model,
    param_dist: dict,
    X_train,
    y_train,
    n_iter: int = 30,
    scoring: str = "roc_auc",
) -> RandomizedSearchCV:
    """Run RandomizedSearchCV and return the fitted searcher."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rscv = RandomizedSearchCV(
        model, param_dist, n_iter=n_iter, cv=cv,
        scoring=scoring, n_jobs=-1, random_state=42, verbose=1,
    )
    rscv.fit(X_train, y_train)
    return rscv


# ── Persistence ──────────────────────────────────────────────────

def save_model(model, metadata: dict, models_dir: Path) -> Path:
    """
    Save model .pkl + metadata JSON to models_dir.

    Stores metadata['model_path'] as a path relative to models_dir
    so the registry works across machines.

    Returns absolute path to saved model file.

    Usage:
        from src.models.train import save_model
        path = save_model(best_model, metadata, MODELS_DIR)
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = type(model).__name__
    # For Pipeline, use the final estimator's class name
    if hasattr(model, "steps"):
        model_name = type(model.steps[-1][1]).__name__

    model_path = models_dir / f"best_model_{model_name}_{ts}.pkl"
    meta_path  = models_dir / f"model_metadata_{ts}.json"

    joblib.dump(model, model_path)

    # Store RELATIVE path so registry is portable
    metadata["model_path_relative"] = model_path.name
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    return model_path