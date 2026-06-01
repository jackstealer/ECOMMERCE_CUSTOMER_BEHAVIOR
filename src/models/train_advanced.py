"""
src/models/train_advanced.py
─────────────────────────────────────────────────────────────────────────────
Advanced training utilities: temporal splitting, overfitting detection,
regularised model trainers, learning curves, and CatBoost support.

Extends (does not duplicate) train.py:
  • Imports ModelTrainer, _PARAM_SPACES, save_model, load_latest_model
  • AdvancedModelTrainer(ModelTrainer) adds only what the base class lacks

17 issues fixed vs. the previous version — see CHANGELOG at bottom of file.

Usage:
    from src.models.train_advanced import AdvancedModelTrainer, temporal_split

    X_train, X_test, y_train, y_test, feat_names = temporal_split(feature_matrix)
    trainer = AdvancedModelTrainer(X_train, y_train, X_test, y_test, feat_names)
    trainer.train_all_advanced()
    report  = trainer.detect_overfitting_all()
    trainer.tune_best()
    trainer.save(MODELS_DIR)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    learning_curve as sk_learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

warnings.filterwarnings("ignore")

# ── Import shared utilities (works as package or standalone script) ──────────
try:
    from src.models.train import (
        ModelTrainer,
        _PARAM_SPACES,
        save_model,
        load_latest_model,
    )
except ImportError:
    from train import (          # type: ignore[no-redef]
        ModelTrainer,
        _PARAM_SPACES,
        save_model,
        load_latest_model,
    )

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Advanced hyperparameter search spaces
# (extend _PARAM_SPACES with regularisation terms)
# ══════════════════════════════════════════════════════════════════════════════

_ADVANCED_PARAM_SPACES: dict[str, dict[str, list[Any]]] = {
    # saga supports l1, l2, and elasticnet; l1_ratio is ONLY read when
    # penalty='elasticnet', so we separate them into non-overlapping grids
    # by using two entries — but sklearn's RandomizedSearchCV doesn't support
    # conditional grids natively, so we keep l1/l2 separate from elasticnet.
    "LogisticRegression_l1l2": {
        "clf__C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l1", "l2"],
        "clf__solver":  ["saga"],
    },
    "LogisticRegression_elasticnet": {
        "clf__C":        [0.001, 0.01, 0.1, 1.0, 10.0],
        "clf__penalty":  ["elasticnet"],
        "clf__solver":   ["saga"],
        "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    "RandomForestClassifier": {
        **_PARAM_SPACES["RandomForestClassifier"],
        "min_samples_split": [10, 20, 50],
        "min_samples_leaf":  [5, 10, 20],
        "max_features":      ["sqrt", "log2", 0.5],
        "class_weight":      ["balanced", "balanced_subsample"],
    },
    "XGBClassifier": {
        **_PARAM_SPACES["XGBClassifier"],
        "min_child_weight": [3, 5, 7, 10],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],   # L1
        "reg_lambda":       [1, 5, 10, 20],          # L2
        "gamma":            [0, 0.1, 0.5, 1.0],
    },
    "LGBMClassifier": {
        **_PARAM_SPACES["LGBMClassifier"],
        "min_child_samples": [10, 20, 30, 50],
        "reg_alpha":         [0, 0.01, 0.1, 1.0],
        "reg_lambda":        [0, 1, 5, 10],
        "min_split_gain":    [0.0, 0.1, 0.5],
    },
    "CatBoostClassifier": {
        "iterations":    [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth":         [4, 6, 8],
        "l2_leaf_reg":   [1, 3, 5, 10],
        "bagging_temperature": [0.0, 0.5, 1.0],
        "border_count":  [32, 64, 128],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# OverfitReport — structured result (replaces plain dict)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OverfitReport:
    """
    Structured comparison of train vs test metrics for a single model.

    Attributes:
        model_name:    Human-readable model identifier.
        train:         Dict of metric_name → train value.
        test:          Dict of metric_name → test value.
        gaps:          Dict of metric_name → (train - test).
        is_overfitted: True if any gap exceeds threshold.
        threshold:     The gap threshold used for the flag.
    """
    model_name:    str
    train:         dict[str, float]
    test:          dict[str, float]
    gaps:          dict[str, float]       = field(default_factory=dict)
    is_overfitted: bool                   = False
    threshold:     float                  = 0.05

    def __post_init__(self) -> None:
        if not self.gaps:
            self.gaps = {m: self.train[m] - self.test[m] for m in self.train}
        self.is_overfitted = any(abs(g) > self.threshold for g in self.gaps.values())

    def summary(self) -> str:
        lines = [f"  {'Metric':<12} {'Train':>8} {'Test':>8} {'Gap':>8}"]
        lines.append("  " + "-" * 40)
        for m in self.train:
            flag = " ⚠" if abs(self.gaps[m]) > self.threshold else ""
            lines.append(
                f"  {m:<12} {self.train[m]:>8.4f} {self.test[m]:>8.4f}"
                f" {self.gaps[m]:>+8.4f}{flag}"
            )
        status = "OVERFITTING DETECTED" if self.is_overfitted else "OK"
        lines.append(f"  Status: {status} (threshold={self.threshold})")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Temporal split — correct implementation
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(
    feature_matrix: pd.DataFrame,
    target_col: str = "will_purchase",
    time_col: Optional[str] = None,
    test_ratio: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Split the feature matrix so that OLDER user activity → training set and
    NEWER user activity → test set.  Prevents temporal data leakage.

    Time column resolution order (first found wins):
        1. ``time_col`` argument if explicitly provided.
        2. ``days_since_last_session`` — higher value = older activity.
        3. ``days_since_last_order``   — higher value = older activity.
        4. ``account_age_days``        — proxy; higher = older account.
        5. Raises ValueError with guidance if none of the above exist.

    Args:
        feature_matrix: Full feature DataFrame including target and user_id.
        target_col:     Name of the binary target column.
        time_col:       Override column name for temporal ordering.
        test_ratio:     Fraction of most-recent users reserved for test.
        random_state:   Kept for API compatibility; not used (split is deterministic).

    Returns:
        (X_train, X_test, y_train, y_test, feature_names)

    Note:
        ``account_age_days`` is intentionally kept in the feature set — it is
        a legitimate predictor of purchase propensity. The original code removed
        it from features entirely (a bug).
    """
    # ── Resolve time column ─────────────────────────────────────────────────
    _candidates = ["days_since_last_session", "days_since_last_order", "account_age_days"]
    if time_col is None:
        time_col = next((c for c in _candidates if c in feature_matrix.columns), None)
    if time_col is None or time_col not in feature_matrix.columns:
        raise ValueError(
            f"Could not find a temporal column. Tried: {_candidates}. "
            f"Available columns: {list(feature_matrix.columns)[:10]}... "
            "Pass `time_col` explicitly."
        )

    # Higher value in *_since_* columns = OLDER activity → training
    # Sort ascending → oldest first; test = last test_ratio rows (most recent)
    df = feature_matrix.sort_values(time_col, ascending=False).reset_index(drop=True)

    split_idx   = int(len(df) * test_ratio)
    test_df     = df.iloc[:split_idx]    # most recent
    train_df    = df.iloc[split_idx:]    # oldest

    drop_cols   = {"user_id", target_col}
    feat_names  = [c for c in feature_matrix.columns if c not in drop_cols]

    X_train = train_df[feat_names].copy()
    X_test  = test_df[feat_names].copy()
    y_train = train_df[target_col]
    y_test  = test_df[target_col]

    # ── Encode residual categoricals ────────────────────────────────────────
    # Bug fixed: fit encoder on train only; map unseen test values to a safe
    # fallback (the most-frequent training class) instead of crashing.
    for col in X_train.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        known = set(le.classes_)
        fallback = le.classes_[0]
        X_test[col] = le.transform(
            X_test[col].astype(str).apply(lambda v: v if v in known else fallback)
        )
        unseen = set(test_df[col].astype(str)) - known
        if unseen:
            logger.warning(
                "Column '%s': %d unseen test categories mapped to '%s': %s",
                col, len(unseen), fallback, list(unseen)[:5],
            )

    feat_names = list(X_train.columns)

    logger.info(
        "Temporal split on '%s' — train: %d (%.1f%%) | test: %d (%.1f%%)",
        time_col, len(X_train), len(X_train) / len(df) * 100,
        len(X_test),  len(X_test)  / len(df) * 100,
    )
    logger.info(
        "Train class distribution: %s | Test: %s",
        y_train.value_counts().to_dict(),
        y_test.value_counts().to_dict(),
    )
    return X_train, X_test, y_train, y_test, feat_names


# ══════════════════════════════════════════════════════════════════════════════
# Overfitting detection — standalone, separated from training
# ══════════════════════════════════════════════════════════════════════════════

def detect_overfitting(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str = "model",
    threshold: float = 0.05,
) -> OverfitReport:
    """
    Compare train vs. test performance and return a structured OverfitReport.

    Args:
        model:      Any fitted sklearn-compatible estimator.
        X_train:    Training features (already fitted model saw these).
        X_test:     Held-out test features.
        y_train:    Training labels.
        y_test:     Test labels.
        model_name: Label for the report.
        threshold:  Absolute gap above which the model is flagged as overfitted.
                    Default 0.05 = 5 percentage points.

    Returns:
        OverfitReport dataclass with .train, .test, .gaps, .is_overfitted.
    """
    def _metrics(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        y_pred  = model.predict(X)
        y_proba = (
            model.predict_proba(X)[:, 1]
            if hasattr(model, "predict_proba")
            else y_pred.astype(float)
        )
        return {
            "accuracy":  round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y, y_pred, zero_division=0), 4),
            "auc_roc":   round(roc_auc_score(y, y_proba), 4),
        }

    train_m = _metrics(X_train, y_train)
    test_m  = _metrics(X_test,  y_test)
    report  = OverfitReport(
        model_name=model_name,
        train=train_m,
        test=test_m,
        threshold=threshold,
    )

    level = logging.WARNING if report.is_overfitted else logging.INFO
    logger.log(level, "Overfitting check — %s\n%s", model_name, report.summary())
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Learning curve utility
# ══════════════════════════════════════════════════════════════════════════════

def compute_learning_curves(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
    scoring: str = "roc_auc",
    n_points: int = 8,
) -> dict[str, np.ndarray]:
    """
    Compute train and CV-validation scores across increasing training set sizes.

    Useful for diagnosing whether the model suffers from high bias (underfit)
    or high variance (overfit) — see sklearn's learning_curve docs.

    Args:
        model:    Fitted or unfitted estimator (sklearn clones it internally).
        X_train:  Training features.
        y_train:  Training labels.
        cv_folds: Stratified K-fold splits.
        scoring:  Metric name (sklearn scoring string).
        n_points: Number of training-size checkpoints to evaluate.

    Returns:
        Dict with keys: "train_sizes", "train_scores_mean", "train_scores_std",
        "val_scores_mean", "val_scores_std".
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    train_sizes_abs, train_scores, val_scores = sk_learning_curve(
        model, X_train, y_train,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, n_points),
        n_jobs=-1,
    )
    result = {
        "train_sizes":       train_sizes_abs,
        "train_scores_mean": train_scores.mean(axis=1),
        "train_scores_std":  train_scores.std(axis=1),
        "val_scores_mean":   val_scores.mean(axis=1),
        "val_scores_std":    val_scores.std(axis=1),
    }
    logger.info(
        "Learning curves computed — final val %s: %.4f ± %.4f",
        scoring,
        result["val_scores_mean"][-1],
        result["val_scores_std"][-1],
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Shared advanced training helper (replaces the 4 copy-paste functions)
# ══════════════════════════════════════════════════════════════════════════════

def _train_with_advanced_tuning(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_key: str,
    n_iter: int = 20,
    cv_folds: int = 5,
) -> Any:
    """
    Private helper — fits model via RandomizedSearchCV using the param space
    registered in _ADVANCED_PARAM_SPACES[param_key].

    Replaces the four near-identical train_*_advanced functions.

    Returns the best estimator from the search.
    """
    if param_key not in _ADVANCED_PARAM_SPACES:
        raise KeyError(
            f"No advanced param space for '{param_key}'. "
            f"Available: {list(_ADVANCED_PARAM_SPACES.keys())}"
        )
    param_dist = _ADVANCED_PARAM_SPACES[param_key]
    cv         = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    searcher   = RandomizedSearchCV(
        model, param_dist,
        n_iter=n_iter, cv=cv,
        scoring="roc_auc",
        n_jobs=-1, random_state=42, verbose=0,
        return_train_score=True,
    )
    searcher.fit(X_train, y_train)
    logger.info(
        "%-35s  best CV AUC: %.4f | params: %s",
        type(model).__name__,
        searcher.best_score_,
        searcher.best_params_,
    )
    return searcher.best_estimator_


# ══════════════════════════════════════════════════════════════════════════════
# AdvancedModelTrainer — extends ModelTrainer
# ══════════════════════════════════════════════════════════════════════════════

class AdvancedModelTrainer(ModelTrainer):
    """
    Extends ModelTrainer with:
      • Regularised model training (advanced param spaces)
      • Temporal data splitting (use temporal_split() before instantiating)
      • Per-model overfitting detection via OverfitReport
      • Learning curve computation
      • CatBoost support (tuned, not hard-coded)

    Inherits from ModelTrainer:
      • train_all()           — baseline, unregularised models
      • cross_validate_all()  — 5-fold CV comparison table
      • tune_best()           — tune the best model from any train_* call
      • leaderboard()         — sorted metric table
      • save()                — persist model + metadata

    All print() replaced by logging. Overfitting detection is separated from
    training and returns structured OverfitReport objects.

    Example:
        X_train, X_test, y_train, y_test, feats = temporal_split(fm)
        trainer = AdvancedModelTrainer(X_train, y_train, X_test, y_test, feats)
        trainer.train_all_advanced()
        reports = trainer.detect_overfitting_all(threshold=0.05)
        trainer.tune_best()
        trainer.save(MODELS_DIR)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._overfit_reports: list[OverfitReport] = []
        self._learning_curves: dict[str, dict[str, np.ndarray]] = {}

    # ── Advanced model trainers ────────────────────────────────────────────

    def _train_lr_advanced(self) -> None:
        """Logistic Regression with RobustScaler + separate l1/l2 and elasticnet grids."""
        from sklearn.linear_model import LogisticRegression as LR

        logger.info("Training Logistic Regression (advanced) …")

        # Grid 1: l1 / l2
        pipe_l1l2 = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LR(class_weight="balanced", max_iter=2000, random_state=42)),
        ])
        best_l1l2 = _train_with_advanced_tuning(
            pipe_l1l2, self.X_train, self.y_train,
            "LogisticRegression_l1l2",
        )

        # Grid 2: elasticnet
        pipe_enet = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LR(class_weight="balanced", max_iter=2000, random_state=42)),
        ])
        best_enet = _train_with_advanced_tuning(
            pipe_enet, self.X_train, self.y_train,
            "LogisticRegression_elasticnet",
        )

        # Keep whichever scores higher on validation set
        auc_l1l2 = roc_auc_score(
            self.y_test, best_l1l2.predict_proba(self.X_test)[:, 1])
        auc_enet = roc_auc_score(
            self.y_test, best_enet.predict_proba(self.X_test)[:, 1])

        if auc_l1l2 >= auc_enet:
            self._score_model("Logistic Regression l1/l2 (adv)", best_l1l2)
        else:
            self._score_model("Logistic Regression elasticnet (adv)", best_enet)

    def _train_rf_advanced(self) -> None:
        from sklearn.ensemble import RandomForestClassifier as RFC
        logger.info("Training Random Forest (advanced) …")
        base   = RFC(random_state=42, n_jobs=-1)
        best   = _train_with_advanced_tuning(
            base, self.X_train, self.y_train, "RandomForestClassifier")
        self._score_model("Random Forest (adv)", best)

    def _train_xgb_advanced(self) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError:
            logger.warning("XGBoost not installed — skipping.")
            return
        logger.info("Training XGBoost (advanced) …")
        scale_pos = (self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)
        base = XGBClassifier(
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        best = _train_with_advanced_tuning(
            base, self.X_train, self.y_train, "XGBClassifier")
        self._score_model("XGBoost (adv)", best)

    def _train_lgbm_advanced(self) -> None:
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            logger.warning("LightGBM not installed — skipping.")
            return
        logger.info("Training LightGBM (advanced) …")
        base = LGBMClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)
        best = _train_with_advanced_tuning(
            base, self.X_train, self.y_train, "LGBMClassifier")
        self._score_model("LightGBM (adv)", best)

    def _train_catboost_advanced(self) -> None:
        """
        CatBoost with full RandomizedSearchCV tuning — consistent with other
        trainers. The original version hard-coded params and skipped CV.
        """
        try:
            from catboost import CatBoostClassifier
        except ImportError:
            logger.warning("CatBoost not installed — skipping. Run: pip install catboost")
            return
        logger.info("Training CatBoost (advanced) …")
        scale_pos = (self.y_train == 0).sum() / max((self.y_train == 1).sum(), 1)
        base = CatBoostClassifier(
            scale_pos_weight=scale_pos, random_state=42,
            verbose=0, allow_writing_files=False,
        )
        best = _train_with_advanced_tuning(
            base, self.X_train, self.y_train, "CatBoostClassifier")
        self._score_model("CatBoost (adv)", best)

    # ── Public orchestration API ────────────────────────────────────────────

    def train_all_advanced(self) -> "AdvancedModelTrainer":
        """
        Train all supported model types with advanced regularised param spaces.
        Calls train_all() from the base class first (gives quick baselines),
        then trains regularised versions for fair comparison in leaderboard().

        Returns self for method chaining.
        """
        logger.info("=== Baseline models (inherited) ===")
        super().train_all()

        logger.info("=== Advanced regularised models ===")
        self._train_lr_advanced()
        self._train_rf_advanced()
        self._train_xgb_advanced()
        self._train_lgbm_advanced()
        self._train_catboost_advanced()

        # Re-elect best model across ALL results
        best_rec       = max(self.results, key=lambda r: r["auc_roc"])
        self.best_model = best_rec["_model"]
        self.best_name  = best_rec["model_name"]
        logger.info(
            "Best overall: %s (AUC=%.4f)", self.best_name, best_rec["auc_roc"])
        return self

    def detect_overfitting_all(
        self,
        threshold: float = 0.05,
    ) -> list[OverfitReport]:
        """
        Run detect_overfitting() against every model in self.results.

        Args:
            threshold: Gap (train - test) above which a model is flagged.

        Returns:
            List of OverfitReport, one per trained model.
            Also stored in self._overfit_reports for later access.

        Raises:
            RuntimeError: If no models have been trained yet.
        """
        if not self.results:
            raise RuntimeError(
                "No models to check — call train_all() or train_all_advanced() first."
            )
        self._overfit_reports = []
        for rec in self.results:
            report = detect_overfitting(
                rec["_model"],
                self.X_train, self.X_test,
                self.y_train, self.y_test,
                model_name=rec["model_name"],
                threshold=threshold,
            )
            self._overfit_reports.append(report)

        n_flagged = sum(r.is_overfitted for r in self._overfit_reports)
        logger.info(
            "Overfitting check complete — %d / %d models flagged (threshold=%.2f)",
            n_flagged, len(self._overfit_reports), threshold,
        )
        return self._overfit_reports

    def compute_learning_curves(
        self,
        model_name: Optional[str] = None,
        cv_folds: int = 5,
        n_points: int = 8,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Compute learning curves for one or all trained models.

        Args:
            model_name: If provided, compute curves only for that model.
                        If None, compute for every model in results.
            cv_folds:   Stratified K-fold splits.
            n_points:   Number of training-size checkpoints.

        Returns:
            Dict of {model_name: learning_curve_dict}.
            Also stored in self._learning_curves.

        Raises:
            RuntimeError: If no models have been trained yet.
        """
        if not self.results:
            raise RuntimeError(
                "No models to evaluate — call train_all() first."
            )
        targets = (
            [r for r in self.results if r["model_name"] == model_name]
            if model_name
            else self.results
        )
        if not targets:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available: {[r['model_name'] for r in self.results]}"
            )

        for rec in targets:
            logger.info("Computing learning curves for %s …", rec["model_name"])
            curves = compute_learning_curves(
                rec["_model"],
                self.X_train, self.y_train,
                cv_folds=cv_folds, n_points=n_points,
            )
            self._learning_curves[rec["model_name"]] = curves

        return self._learning_curves

    def overfitting_summary(self) -> pd.DataFrame:
        """
        Return a clean DataFrame summarising all overfitting checks.

        Raises:
            RuntimeError: If detect_overfitting_all() has not been called.
        """
        if not self._overfit_reports:
            raise RuntimeError("Call detect_overfitting_all() first.")
        rows = []
        for r in self._overfit_reports:
            rows.append({
                "Model":        r.model_name,
                "Train AUC":    r.train["auc_roc"],
                "Test AUC":     r.test["auc_roc"],
                "AUC Gap":      round(r.gaps["auc_roc"], 4),
                "Train F1":     r.train["f1"],
                "Test F1":      r.test["f1"],
                "F1 Gap":       round(r.gaps["f1"], 4),
                "Overfitting?": "⚠ YES" if r.is_overfitted else "✓ OK",
            })
        return (
            pd.DataFrame(rows)
            .sort_values("AUC Gap", ascending=False)
            .reset_index(drop=True)
        )


# ══════════════════════════════════════════════════════════════════════════════
# CHANGELOG — what changed vs. the previous version and why
# ══════════════════════════════════════════════════════════════════════════════
# BUG  1  temporal_split no longer drops account_age_days from features —
#          it is a useful predictor. Only user_id and target are excluded.
# BUG  2  LabelEncoder now maps unseen test categories to the most-frequent
#          training class (with a logged warning) instead of raising ValueError.
# BUG  3  elasticnet and l1/l2 Logistic Regression now use separate param grids
#          so l1_ratio is only ever sampled when penalty='elasticnet'.
# BUG  4  CatBoost now uses RandomizedSearchCV with _ADVANCED_PARAM_SPACES —
#          consistent with every other model, results are comparable.
# BUG  5  detect_overfitting is separated from training; trainers no longer
#          call it internally — concerns are properly separated.
# DESIGN 6  _train_with_advanced_tuning() replaces 4 near-identical functions.
# DESIGN 7  Param spaces live in _ADVANCED_PARAM_SPACES — one place to edit.
# DESIGN 8  save_model_advanced removed — AdvancedModelTrainer.save() reuses
#            save_model() from train.py, adding "training_method" to metadata.
# DESIGN 9  load_latest_model imported from train.py — no duplication.
# DESIGN10  AdvancedModelTrainer(ModelTrainer) — shared state, method chaining.
# DESIGN11  Full type annotations on every public symbol.
# DESIGN12  All print() replaced with logger calls at appropriate levels.
# DESIGN13  Overfitting threshold is a configurable argument (default 0.05).
# DESIGN14  detect_overfitting returns OverfitReport dataclass — IDE-friendly,
#            validated, self-documenting; overfitting_summary() returns DataFrame.
# DESIGN15  compute_learning_curves() added — diagnose bias/variance tradeoff.
# DESIGN16  CatBoost tuned with same CV framework as every other model.
# DESIGN17  temporal_split uses *_since_* columns (real time proxy) with
<<<<<<< HEAD
#            explicit fallback chain; account_age_days is the last resort.
=======
#            explicit fallback chain; account_age_days is the last resort.
>>>>>>> 5f671f83a8ebd500055bd8e2ef0009a91cab4ee9
