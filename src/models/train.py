"""
src/models/train.py
─────────────────────────────────────────────────────────────────────────────
Production-grade model training utilities.

Key improvements over the previous version
  ✓ Logging everywhere (no bare print statements)
  ✓ Full type annotations (Python 3.9-compatible)
  ✓ ModelTrainer class — encapsulates the full train → tune → save workflow
  ✓ Model registry / factory — add new models in one place
  ✓ All hyperparameter search spaces defined centrally (no repetition)
  ✓ Cross-validation results stored alongside test metrics
  ✓ save_model stores feature names so downstream code never guesses
  ✓ load_model / load_latest_model helpers for inference scripts
  ✓ Graceful optional-dependency handling (XGBoost, LightGBM)
  ✓ prepare_data correctly label-encodes residual categoricals

Usage — scripted pipeline:
    from src.models.train import ModelTrainer
    trainer = ModelTrainer(X_train, y_train, X_test, y_test, feature_names)
    trainer.train_all()
    trainer.tune_best()
    path = trainer.save()

Usage — individual helpers:
    from src.models.train import prepare_data, train_lightgbm, save_model
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# ── Module-level logger ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Central hyperparameter search spaces ────────────────────────────────────
_PARAM_SPACES: dict[str, dict[str, list[Any]]] = {
    "LGBMClassifier": {
        "n_estimators":      [100, 200, 300, 500],
        "learning_rate":     [0.01, 0.03, 0.05, 0.1],
        "max_depth":         [3, 5, 6, 8, -1],
        "num_leaves":        [20, 31, 40, 60],
        "min_child_samples": [10, 20, 30, 50],
        "subsample":         [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
    },
    "XGBClassifier": {
        "n_estimators":  [100, 200, 300, 500],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth":     [3, 4, 5, 6, 8],
        "subsample":     [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "gamma":         [0, 0.1, 0.2, 0.5],
    },
    "RandomForestClassifier": {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [5, 8, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2"],
    },
    "LogisticRegression": {
        "clf__C":        [0.001, 0.01, 0.1, 1.0, 10.0],
        "clf__penalty":  ["l1", "l2"],
        "clf__solver":   ["liblinear", "saga"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Data preparation
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(
    feature_matrix: pd.DataFrame,
    target_col: str = "will_purchase",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Split feature matrix into stratified train/test sets.

    Automatically label-encodes any residual object/category columns so
    every downstream model receives a fully numeric DataFrame.

    Args:
        feature_matrix: Full feature DataFrame including target and user_id.
        target_col:     Name of the binary target column.
        test_size:      Fraction reserved for the test set.
        random_state:   Reproducibility seed — must match ModelConfig.RANDOM_STATE.

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)

    Raises:
        ValueError: If target_col is missing from feature_matrix.
        ValueError: If feature_matrix has fewer than 2 distinct target values.
    """
    if target_col not in feature_matrix.columns:
        raise ValueError(
            f"Target column '{target_col}' not found. "
            f"Available: {list(feature_matrix.columns)}"
        )

    n_classes = feature_matrix[target_col].nunique()
    if n_classes < 2:
        raise ValueError(
            f"Target '{target_col}' has only {n_classes} unique value(s). "
            "Need at least 2 for classification."
        )

    drop_cols = {target_col, "user_id"}
    feature_names = [c for c in feature_matrix.columns if c not in drop_cols]

    X = feature_matrix[feature_names].copy()

    # Encode any residual string / categorical columns
    encoded: list[str] = []
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        encoded.append(col)
    if encoded:
        logger.warning(
            "Label-encoded %d residual string/category column(s): %s. "
            "Consider encoding these in Phase 2 instead.",
            len(encoded),
            encoded,
        )

    feature_names = list(X.columns)
    y = feature_matrix[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logger.info(
        "Data split — train: %d | test: %d | positive rate train: %.1f%% | test: %.1f%%",
        len(X_train), len(X_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test, feature_names


# ══════════════════════════════════════════════════════════════════════════════
# Class weight helper
# ══════════════════════════════════════════════════════════════════════════════

def get_class_weights(y: pd.Series) -> dict[int, float]:
    """
    Compute class weights inversely proportional to class frequency.

    Returns:
        {0: weight_negative, 1: weight_positive}
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    cw = dict(zip(classes.tolist(), weights.tolist()))
    logger.debug("Class weights: %s", cw)
    return cw


# ══════════════════════════════════════════════════════════════════════════════
# Individual model trainers
# ══════════════════════════════════════════════════════════════════════════════

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> Pipeline:
    """
    Fit Logistic Regression inside a StandardScaler Pipeline.

    Returns a sklearn Pipeline so it can be saved and loaded without
    a separate scaler artefact.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            **kwargs,
        )),
    ])
    pipe.fit(X_train, y_train)
    logger.info("Logistic Regression trained.")
    return pipe


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Fit a balanced Random Forest and return the fitted model."""
    params: dict[str, Any] = dict(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    params.update(kwargs)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("Random Forest trained (n_estimators=%d).", params["n_estimators"])
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> Optional[Any]:
    """
    Fit XGBoost with automatic scale_pos_weight for class imbalance.

    Returns None (with a log warning) if xgboost is not installed.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.warning("XGBoost not installed — skipping. Run: pip install xgboost")
        return None

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params: dict[str, Any] = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    params.update(kwargs)
    # Remove kwarg removed in XGBoost ≥1.6 — prevents TypeError on newer installs
    params.pop("use_label_encoder", None)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("XGBoost trained (scale_pos_weight=%.2f).", scale_pos)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs: Any,
) -> Optional[Any]:
    """
    Fit LightGBM with balanced class weights.

    Returns None (with a log warning) if lightgbm is not installed.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.warning("LightGBM not installed — skipping. Run: pip install lightgbm")
        return None

    params: dict[str, Any] = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=40,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(kwargs)
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    logger.info("LightGBM trained.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter tuning
# ══════════════════════════════════════════════════════════════════════════════

def tune_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_dist: Optional[dict[str, list[Any]]] = None,
    n_iter: int = 30,
    scoring: str = "roc_auc",
    cv_folds: int = 5,
) -> RandomizedSearchCV:
    """
    Run RandomizedSearchCV on model.

    If param_dist is None, looks up the central _PARAM_SPACES registry
    using the model's class name — so you never have to pass a param grid
    manually for supported model types.

    Args:
        model:      Unfitted estimator (or Pipeline) to tune.
        X_train:    Training features.
        y_train:    Training labels.
        param_dist: Override parameter distribution. If None, uses _PARAM_SPACES.
        n_iter:     Number of random configurations to try.
        scoring:    Metric to optimise (default: roc_auc).
        cv_folds:   Number of stratified CV folds.

    Returns:
        Fitted RandomizedSearchCV object. Access .best_estimator_ for the model.

    Raises:
        KeyError: If param_dist is None and model type not in _PARAM_SPACES.
    """
    model_class = type(model).__name__

    if param_dist is None:
        if model_class not in _PARAM_SPACES:
            raise KeyError(
                f"No default param space for '{model_class}'. "
                f"Pass param_dist explicitly or add to _PARAM_SPACES. "
                f"Supported: {list(_PARAM_SPACES.keys())}"
            )
        param_dist = _PARAM_SPACES[model_class]

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rscv = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=42,
        verbose=1,
        return_train_score=True,
    )
    rscv.fit(X_train, y_train)

    logger.info(
        "Tuning complete — best CV %s: %.4f | params: %s",
        scoring,
        rscv.best_score_,
        rscv.best_params_,
    )
    return rscv


# ══════════════════════════════════════════════════════════════════════════════
# Persistence helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_model(
    model: Any,
    metadata: dict[str, Any],
    models_dir: Path,
    feature_names: Optional[list[str]] = None,
) -> Path:
    """
    Save model .pkl and metadata JSON to models_dir.

    Stores the model filename as a relative path inside metadata so the
    registry works correctly when the project is cloned on another machine.

    Args:
        model:         Fitted estimator or Pipeline.
        metadata:      Dict of metrics / params to persist alongside the model.
        models_dir:    Directory where artefacts are written.
        feature_names: If provided, stored in metadata for inference-time validation.

    Returns:
        Absolute path to the saved .pkl file.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # Resolve display name (Pipeline → inner estimator class)
    if hasattr(model, "steps"):
        display_name = type(model.steps[-1][1]).__name__
    else:
        display_name = type(model).__name__

    model_path = models_dir / f"best_model_{display_name}_{ts}.pkl"
    meta_path  = models_dir / f"model_metadata_{ts}.json"

    joblib.dump(model, model_path)

    meta_out = {
        **metadata,
        "model_class":    display_name,
        "trained_at":     ts,
        "model_filename": model_path.name,   # relative — not absolute
    }
    if feature_names:
        meta_out["feature_names"] = feature_names

    meta_path.write_text(json.dumps(meta_out, indent=2, default=str))

    logger.info("Model saved → %s", model_path.name)
    logger.info("Metadata saved → %s", meta_path.name)
    return model_path


def load_latest_model(models_dir: Path) -> tuple[Any, dict[str, Any]]:
    """
    Load the most recently saved model and its metadata.

    Args:
        models_dir: Directory containing best_model_*.pkl files.

    Returns:
        (fitted_model, metadata_dict)

    Raises:
        FileNotFoundError: If no model files exist in models_dir.
    """
    model_files = sorted(models_dir.glob("best_model_*.pkl"))
    if not model_files:
        raise FileNotFoundError(
            f"No saved model found in {models_dir}. "
            "Run the training pipeline (Phase 5) first."
        )
    model_path = model_files[-1]
    model = joblib.load(model_path)

    # Load matching metadata
    ts         = model_path.stem.rsplit("_", 1)[-1]          # YYYYMMDD_HHMM
    meta_candidates = sorted(models_dir.glob(f"model_metadata_{ts}.json"))
    metadata = (
        json.loads(meta_candidates[0].read_text())
        if meta_candidates
        else {}
    )

    logger.info("Loaded model: %s", model_path.name)
    return model, metadata


# ══════════════════════════════════════════════════════════════════════════════
# High-level ModelTrainer class
# ══════════════════════════════════════════════════════════════════════════════

class ModelTrainer:
    """
    Orchestrates the full train → compare → tune → save workflow.

    Attributes:
        results (list[dict]): Performance metrics for every trained model.
        best_model:           The model with the highest test-set AUC after tuning.
        best_name (str):      Human-readable name of best_model.

    Example:
        trainer = ModelTrainer(X_train, y_train, X_test, y_test, feature_names)
        trainer.train_all()
        trainer.tune_best()
        path = trainer.save(models_dir=MODELS_DIR)
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str],
        random_state: int = 42,
    ) -> None:
        self.X_train       = X_train
        self.y_train       = y_train
        self.X_test        = X_test
        self.y_test        = y_test
        self.feature_names = feature_names
        self.random_state  = random_state
        self.results:      list[dict[str, Any]] = []
        self.best_model:   Optional[Any]        = None
        self.best_name:    str                  = ""
        self._tuner:       Optional[RandomizedSearchCV] = None

    # ── private helpers ────────────────────────────────────────────────────

    def _score_model(self, name: str, model: Any) -> dict[str, Any]:
        """Fit model on train, evaluate on test, store in self.results."""
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
        )

        model.fit(self.X_train, self.y_train)
        y_pred  = model.predict(self.X_test)
        y_proba = (
            model.predict_proba(self.X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred.astype(float)
        )
        record = {
            "model_name": name,
            "accuracy":   round(accuracy_score(self.y_test, y_pred),                4),
            "precision":  round(precision_score(self.y_test, y_pred, zero_division=0), 4),
            "recall":     round(recall_score(self.y_test, y_pred, zero_division=0),  4),
            "f1":         round(f1_score(self.y_test, y_pred, zero_division=0),      4),
            "auc_roc":    round(roc_auc_score(self.y_test, y_proba),                 4),
            "_model":     model,
            "_proba":     y_proba,
        }
        self.results.append(record)
        logger.info(
            "%-30s  Acc:%.4f  F1:%.4f  AUC:%.4f",
            name, record["accuracy"], record["f1"], record["auc_roc"],
        )
        return record

    # ── public API ─────────────────────────────────────────────────────────

    def train_all(self) -> "ModelTrainer":
        """
        Train all supported model types and store results.

        Skips XGBoost / LightGBM gracefully if not installed.
        Returns self for method chaining.
        """
        logger.info("=== Training baseline models ===")
        self._score_model("Logistic Regression",
                          train_logistic_regression(self.X_train, self.y_train))
        self._score_model("Random Forest",
                          train_random_forest(self.X_train, self.y_train))

        logger.info("=== Training advanced models ===")
        xgb = train_xgboost(self.X_train, self.y_train)
        if xgb:
            self._score_model("XGBoost", xgb)

        lgbm = train_lightgbm(self.X_train, self.y_train)
        if lgbm:
            self._score_model("LightGBM", lgbm)

        # Identify provisional best by test AUC
        best_rec    = max(self.results, key=lambda r: r["auc_roc"])
        self.best_model = best_rec["_model"]
        self.best_name  = best_rec["model_name"]
        logger.info("Provisional best: %s (AUC=%.4f)", self.best_name, best_rec["auc_roc"])
        return self

    def cross_validate_all(self, cv_folds: int = 5) -> pd.DataFrame:
        """
        Run stratified cross-validation on every trained model.

        Returns:
            DataFrame with mean ± std of F1 and AUC for each model.
        """
        cv  = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        rows: list[dict[str, str]] = []
        for rec in self.results:
            scores = cross_validate(
                rec["_model"], self.X_train, self.y_train,
                cv=cv, scoring=["f1", "roc_auc"], n_jobs=-1,
            )
            rows.append({
                "Model":   rec["model_name"],
                "CV F1":   f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
                "CV AUC":  f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
            })
            logger.info(
                "%-30s  CV AUC: %.4f ± %.4f",
                rec["model_name"],
                scores["test_roc_auc"].mean(),
                scores["test_roc_auc"].std(),
            )
        return pd.DataFrame(rows)

    def tune_best(
        self,
        n_iter: int = 30,
        param_dist: Optional[dict[str, list[Any]]] = None,
    ) -> "ModelTrainer":
        """
        Tune the best model found in train_all() using RandomizedSearchCV.

        Automatically looks up the param space from _PARAM_SPACES unless
        param_dist is provided explicitly.

        Returns self for method chaining.

        Raises:
            RuntimeError: If train_all() has not been called yet.
        """
        if self.best_model is None:
            raise RuntimeError("Call train_all() before tune_best().")

        model_class = (
            type(self.best_model.steps[-1][1]).__name__
            if hasattr(self.best_model, "steps")
            else type(self.best_model).__name__
        )
        logger.info("=== Tuning %s ===", model_class)

        self._tuner = tune_model(
            self.best_model,
            self.X_train,
            self.y_train,
            param_dist=param_dist,
            n_iter=n_iter,
        )
        self.best_model = self._tuner.best_estimator_

        # Re-evaluate tuned model
        self._score_model(f"{self.best_name} (tuned)", self.best_model)
        tuned_rec = self.results[-1]
        logger.info(
            "Tuned AUC on test set: %.4f", tuned_rec["auc_roc"]
        )
        return self

    def leaderboard(self) -> pd.DataFrame:
        """Return a clean comparison table of all trained models, sorted by AUC."""
        df = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in self.results
        ]).sort_values("auc_roc", ascending=False).reset_index(drop=True)
        df.index += 1
        return df

    def save(
        self,
        models_dir: Path,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> Path:
        """
        Persist the current best_model + metadata to models_dir.

        Metadata includes: model class, best params (if tuned),
        CV AUC, test AUC/F1, n_features, feature_names, trained_at.

        Returns:
            Absolute path to the saved .pkl file.

        Raises:
            RuntimeError: If no model has been trained yet.
        """
        if self.best_model is None:
            raise RuntimeError("No model to save — call train_all() first.")

        tuned_rec = next(
            (r for r in reversed(self.results) if "(tuned)" in r["model_name"]),
            self.results[-1],
        )

        metadata: dict[str, Any] = {
            "model_type":     self.best_name,
            "test_auc":       tuned_rec["auc_roc"],
            "test_f1":        tuned_rec["f1"],
            "test_precision": tuned_rec["precision"],
            "test_recall":    tuned_rec["recall"],
            "test_accuracy":  tuned_rec["accuracy"],
            "cv_auc":         round(self._tuner.best_score_, 4) if self._tuner else None,
            "best_params":    self._tuner.best_params_ if self._tuner else {},
            "n_features":     len(self.feature_names),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return save_model(
            self.best_model,
            metadata,
            models_dir,
            feature_names=self.feature_names,
        )
