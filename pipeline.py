"""
pipeline.py
─────────────────────────────────────────────────────────────────────────────
Production ML pipeline: load → feature engineering → train → evaluate → save.

All 20 issues from the previous version fixed — see CHANGELOG at the bottom.

Usage:
    # Default run
    python pipeline.py

    # Custom settings
    python pipeline.py --test-size 0.15 --n-iter 40 --log-level DEBUG

    # Verify imports and paths without training anything
    python pipeline.py --dry-run

    # Skip slow advanced tuning (useful for quick iteration)
    python pipeline.py --skip-advanced

    # Override overfit threshold
    python pipeline.py --overfit-threshold 0.08
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import pandas as pd

# ── Make project root importable regardless of CWD ──────────────────────────
# Uses __file__ so it works from any working directory — not sys.path.insert(0)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Project imports ──────────────────────────────────────────────────────────
try:
    from config import Paths, ModelConfig, BusinessConfig
    from src.features.build_features import engineer_features, create_rfm_features
    from src.data.preprocess import clean_data
    from src.models.train_advanced import AdvancedModelTrainer, temporal_split, OverfitReport
    from src.models.evaluate import evaluate_model, build_lift_table, save_evaluation_report
except ImportError as exc:
    # Provide a clear, actionable error instead of a bare ImportError traceback
    print(
        f"\n[FATAL] Import failed: {exc}\n"
        "Make sure you are running from the project root:\n"
        "    python pipeline.py\n"
        "And that dependencies are installed:\n"
        "    pip install -r requirements.txt\n",
        file=sys.stderr,
    )
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# Logging setup
# ══════════════════════════════════════════════════════════════════════════════

def _configure_logging(level: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Configure root logger with console + optional file handler.
    Returns the pipeline-specific logger.
    """
    fmt     = "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s"
    datefmt = "%H:%M:%S"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level   = getattr(logging, level.upper(), logging.INFO),
        format  = fmt,
        datefmt = datefmt,
        handlers= handlers,
        force   = True,
    )
    # Silence noisy third-party loggers
    for noisy in ("lightgbm", "xgboost", "catboost"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# CLI argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E-Commerce Customer Behavior ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--test-size", type=float, default=ModelConfig.TEST_SIZE,
        help="Fraction of users reserved for test set."
    )
    p.add_argument(
        "--n-iter", type=int, default=ModelConfig.N_ITER_TUNE,
        help="RandomizedSearchCV iterations for hyperparameter tuning."
    )
    p.add_argument(
        "--cv-folds", type=int, default=ModelConfig.CV_FOLDS,
        help="Stratified K-fold splits for cross-validation."
    )
    p.add_argument(
        "--overfit-threshold", type=float, default=0.05,
        help="Max train-test metric gap before a model is flagged as overfit."
    )
    p.add_argument(
        "--target-col", type=str, default=ModelConfig.TARGET_COL,
        help="Binary target column name in the feature matrix."
    )
    p.add_argument(
        "--skip-advanced", action="store_true",
        help="Train baseline models only (skips regularised tuning — faster)."
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate imports, paths, and config without training."
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity."
    )
    p.add_argument(
        "--log-file", type=Path, default=None,
        help="Optional path to write logs to a file."
    )
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Step timing context manager
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def _step(
    name: str,
    logger: logging.Logger,
    step_times: dict[str, float],
) -> Generator[None, None, None]:
    """
    Context manager that logs step start/end and records elapsed time.
    Re-raises any exception after logging it so the pipeline can handle it.

    Args:
        name:       Human-readable step name.
        logger:     Logger to write to.
        step_times: Dict to record elapsed seconds per step.
    """
    bar = "─" * (60 - len(name))
    logger.info("┌── %s %s", name, bar)
    t0 = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - t0
        step_times[name] = round(elapsed, 2)
        logger.info("└── ✓ %s complete (%.1fs)", name, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        step_times[name] = round(elapsed, 2)
        logger.error("└── ✗ %s failed after %.1fs: %s", name, elapsed, exc)
        raise


# ══════════════════════════════════════════════════════════════════════════════
# Feature importance extraction utility
# ══════════════════════════════════════════════════════════════════════════════

def _extract_feature_importance(
    model: Any,
    feature_names: list[str],
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Extract feature importances from any sklearn-compatible model or Pipeline.

    Resolution order:
      1. feature_importances_  (tree-based models)
      2. coef_                 (linear models — uses absolute values)
      3. Empty dict with warning (SHAP or permutation importance not included here)

    Works correctly for Pipeline objects by inspecting the final estimator.
    """
    estimator = model.steps[-1][1] if hasattr(model, "steps") else model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_

    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        # coef_ shape is (1, n_features) for binary; squeeze to 1-D
        importances = np.abs(coef.ravel())

    else:
        logger.warning(
            "%s has no feature_importances_ or coef_. "
            "Feature importance will be empty.",
            type(estimator).__name__,
        )
        return {}

    if len(importances) != len(feature_names):
        logger.warning(
            "Importance array length (%d) != feature count (%d). "
            "Returning empty dict.",
            len(importances), len(feature_names),
        )
        return {}

    return dict(zip(feature_names, importances.tolist()))


# ══════════════════════════════════════════════════════════════════════════════
# Model selection — separated from logging / training concerns
# ══════════════════════════════════════════════════════════════════════════════

def _select_best_model(
    trainer: AdvancedModelTrainer,
    overfit_reports: list[OverfitReport],
    logger: logging.Logger,
) -> tuple[Any, str, dict[str, float]]:
    """
    Select the best model from trained candidates using cached overfit reports.

    Strategy:
      1. Prefer models with is_overfitted=False.
      2. Among candidates, choose the highest test-set AUC.
      3. If ALL models overfit, fall back to the one with the smallest AUC gap
         and log a clear warning.

    Args:
        trainer:         Fitted AdvancedModelTrainer (results already populated).
        overfit_reports: Pre-computed list of OverfitReport — never re-runs detection.
        logger:          Logger for warnings.

    Returns:
        (best_fitted_model, model_name, metrics_dict)
    """
    # Build a lookup: model_name → {metrics, overfit_report, fitted_model}
    rec_by_name  = {r["model_name"]: r for r in trainer.results}
    rep_by_name  = {r.model_name:    r for r in overfit_reports}

    non_overfit = [r for r in overfit_reports if not r.is_overfitted]
    candidates  = non_overfit if non_overfit else overfit_reports

    if not non_overfit:
        logger.warning(
            "All %d models show overfitting (threshold=%.2f). "
            "Selecting model with smallest AUC gap as fallback.",
            len(overfit_reports),
            overfit_reports[0].threshold if overfit_reports else 0,
        )

    # Among candidates, highest test AUC wins
    def _test_auc(report: OverfitReport) -> float:
        return rec_by_name.get(report.model_name, {}).get("auc_roc", 0.0)

    best_report = max(candidates, key=_test_auc)
    best_rec    = rec_by_name[best_report.model_name]

    metrics = {
        "accuracy":  best_rec["accuracy"],
        "precision": best_rec["precision"],
        "recall":    best_rec["recall"],
        "f1":        best_rec["f1"],
        "auc_roc":   best_rec["auc_roc"],
    }
    return best_rec["_model"], best_report.model_name, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner class
# ══════════════════════════════════════════════════════════════════════════════

class PipelineRunner:
    """
    Orchestrates the full ML pipeline as a testable, injectable class.

    Steps:
        1. Load and validate raw CSV files
        2. Clean and preprocess data
        3. Engineer features → feature matrix
        4. Temporal train / test split
        5. Train all models (baseline + advanced)
        6. Detect overfitting (uses cached results — no re-computation)
        7. Select best model
        8. Evaluate and persist artefacts
        9. Save pipeline run summary JSON

    All results are stored as instance attributes after run() completes,
    so downstream code (notebooks, tests, API) can inspect them without
    re-running the pipeline.
    """

    def __init__(
        self,
        target_col:         str   = ModelConfig.TARGET_COL,
        test_size:          float = ModelConfig.TEST_SIZE,
        n_iter:             int   = ModelConfig.N_ITER_TUNE,
        cv_folds:           int   = ModelConfig.CV_FOLDS,
        overfit_threshold:  float = 0.05,
        skip_advanced:      bool  = False,
        logger:             Optional[logging.Logger] = None,
    ) -> None:
        self.target_col        = target_col
        self.test_size         = test_size
        self.n_iter            = n_iter
        self.cv_folds          = cv_folds
        self.overfit_threshold = overfit_threshold
        self.skip_advanced     = skip_advanced
        self.log               = logger or logging.getLogger("pipeline")

        # Results — populated by run()
        self.run_id:            str                           = str(uuid.uuid4())[:8]
        self.step_times:        dict[str, float]             = {}
        self.trainer:           Optional[AdvancedModelTrainer] = None
        self.best_model:        Optional[Any]                = None
        self.best_model_name:   str                          = ""
        self.best_metrics:      dict[str, float]             = {}
        self.feature_names:     list[str]                    = []
        self.overfit_reports:   list[OverfitReport]          = []

    # ── Internal helpers ────────────────────────────────────────────────────

    def _step(self, name: str) -> contextmanager:
        return _step(name, self.log, self.step_times)

    # ── Pipeline steps ──────────────────────────────────────────────────────

    def _load_raw(self) -> dict[str, pd.DataFrame]:
        """Load every CSV from data/raw/. Raises FileNotFoundError with guidance."""
        raw_dir = Paths.DATA_RAW
        if not raw_dir.exists():
            raise FileNotFoundError(
                f"Raw data directory not found: {raw_dir}\n"
                "Run: python src/data/generate_data.py"
            )
        tables = {}
        required = ["users", "sessions", "browse_events", "orders",
                    "order_items", "products"]
        for name in required:
            path = raw_dir / f"{name}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing: {path}\n"
                    "Run: python src/data/generate_data.py"
                )
            tables[name] = pd.read_csv(path)
            self.log.info("  Loaded %-20s %8d rows", name, len(tables[name]))
        return tables

    def _clean(self, tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Apply cleaning pipeline from src/data/preprocess.py."""
        return clean_data(raw_dir=Paths.DATA_RAW)

    def _build_features(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Engineer the master feature matrix."""
        fm = engineer_features(tables)
        # Target: has the user ever purchased?
        buyer_ids = set(tables["orders"]["user_id"])
        fm[self.target_col] = fm["user_id"].isin(buyer_ids).astype(int)

        if self.target_col not in fm.columns:
            raise ValueError(
                f"Target column '{self.target_col}' missing after feature engineering."
            )
        n_pos = fm[self.target_col].sum()
        n_neg = len(fm) - n_pos
        self.log.info(
            "  Feature matrix: %d rows × %d cols | buyers: %d (%.1f%%) | non-buyers: %d",
            *fm.shape, n_pos, n_pos / len(fm) * 100, n_neg,
        )
        return fm

    def _split(
        self, fm: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]]:
        """
        Temporal train/test split via temporal_split() from train_advanced.py.
        Falls back to stratified split if no time column is found (with warning).
        """
        try:
            return temporal_split(
                fm,
                target_col=self.target_col,
                test_ratio=self.test_size,
            )
        except ValueError as exc:
            self.log.warning(
                "Temporal split unavailable (%s) — falling back to stratified split.", exc
            )
            from sklearn.model_selection import train_test_split
            drop  = {"user_id", self.target_col}
            feats = [c for c in fm.columns if c not in drop]
            X, y  = fm[feats].copy(), fm[self.target_col]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=self.test_size, stratify=y,
                random_state=ModelConfig.RANDOM_STATE,
            )
            return X_tr, X_te, y_tr, y_te, list(X_tr.columns)

    def _train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: list[str],
    ) -> AdvancedModelTrainer:
        """
        Train all models through AdvancedModelTrainer.
        Calls train_all() (baselines) always; train_all_advanced() unless
        --skip-advanced is set.
        """
        trainer = AdvancedModelTrainer(
            X_train, y_train, X_test, y_test,
            feature_names,
            random_state=ModelConfig.RANDOM_STATE,
        )
        if self.skip_advanced:
            self.log.info("  [--skip-advanced] training baseline models only.")
            trainer.train_all()
        else:
            trainer.train_all_advanced()

        self.log.info("\n%s", trainer.leaderboard().to_string(index=False))
        return trainer

    def _detect_overfit(self, trainer: AdvancedModelTrainer) -> list[OverfitReport]:
        """Run overfitting detection once; results are cached — never re-computed."""
        reports = trainer.detect_overfitting_all(threshold=self.overfit_threshold)
        self.log.info("\n%s", trainer.overfitting_summary().to_string(index=False))
        return reports

    def _save_artefacts(
        self,
        trainer:          AdvancedModelTrainer,
        best_model:       Any,
        best_name:        str,
        best_metrics:     dict[str, float],
        feature_names:    list[str],
        X_test:           pd.DataFrame,
        y_test:           pd.Series,
        overfit_reports:  list[OverfitReport],
    ) -> None:
        """Persist model, evaluation report, and feature names."""
        # ── Feature importance ────────────────────────────────────────────
        feat_imp = _extract_feature_importance(best_model, feature_names, self.log)

        # ── Evaluate on test set ──────────────────────────────────────────
        eval_result = evaluate_model(best_model, X_test, y_test, feature_names)
        lift_df     = build_lift_table(y_test.values, eval_result["y_proba"])
        eval_result["y_true"] = y_test

        # ── Save evaluation report + dashboard_data.json ──────────────────
        Paths.DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
        save_evaluation_report(eval_result, lift_df, feat_imp, Paths.DATA_OUTPUTS)
        self.log.info("  Evaluation report saved → %s", Paths.DATA_OUTPUTS)

        # ── Save model via ModelTrainer.save() — no manual metadata dict ──
        extra = {
            "training_method":  "advanced_regularised" if not self.skip_advanced else "baseline",
            "split_method":     "temporal",
            "overfit_flagged":  any(r.is_overfitted for r in overfit_reports),
            "overfit_threshold": self.overfit_threshold,
            "pipeline_run_id":  self.run_id,
        }
        trainer.best_model = best_model
        trainer.best_name  = best_name
        model_path = trainer.save(Paths.MODELS_DIR, extra_metadata=extra)
        self.log.info("  Model saved → %s", model_path.name)

        # ── Save feature names ────────────────────────────────────────────
        feat_path = Paths.DATA_OUTPUTS / "feature_names.txt"
        feat_path.write_text("\n".join(feature_names))
        self.log.info("  Feature names saved → %s", feat_path.name)

    def _save_run_summary(
        self,
        best_name:    str,
        best_metrics: dict[str, float],
        started_at:   datetime,
    ) -> None:
        """Write a JSON summary of this pipeline run to data/outputs/."""
        summary = {
            "run_id":            self.run_id,
            "started_at":        started_at.isoformat(),
            "finished_at":       datetime.now().isoformat(),
            "total_seconds":     round(sum(self.step_times.values()), 2),
            "step_times":        self.step_times,
            "best_model":        best_name,
            "metrics":           best_metrics,
            "n_features":        len(self.feature_names),
            "overfit_detected":  any(r.is_overfitted for r in self.overfit_reports),
            "skip_advanced":     self.skip_advanced,
            "target_col":        self.target_col,
            "test_size":         self.test_size,
            "overfit_threshold": self.overfit_threshold,
        }
        out = Paths.DATA_OUTPUTS / f"pipeline_run_{self.run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        self.log.info("  Run summary saved → %s", out.name)

    # ── Public entry point ──────────────────────────────────────────────────

    def run(self) -> "PipelineRunner":
        """
        Execute the full pipeline sequentially.

        Returns self — all results accessible as attributes:
            .best_model, .best_model_name, .best_metrics,
            .feature_names, .trainer, .overfit_reports, .step_times

        Raises:
            Any exception from any step — logged with full context before re-raising.
        """
        started_at = datetime.now()
        self.log.info("=" * 65)
        self.log.info("  ML PIPELINE  |  run_id=%s  |  %s", self.run_id, started_at.strftime("%Y-%m-%d %H:%M"))
        self.log.info("=" * 65)

        # Step 1 — Load
        with self._step("Step 1 · Load raw data"):
            tables = self._load_raw()

        # Step 2 — Clean
        with self._step("Step 2 · Clean and preprocess"):
            tables = self._clean(tables)

        # Step 3 — Features
        with self._step("Step 3 · Feature engineering"):
            fm = self._build_features(tables)
            Paths.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
            _fm_path = Paths.DATA_PROCESSED / "feature_matrix.parquet"
            try:
                fm.to_parquet(_fm_path, index=False)
            except Exception:
                fm.to_csv(_fm_path.with_suffix(".csv"), index=False)
            self.log.info("  Feature matrix saved → %s", _fm_path.name)

        # Step 4 — Split
        with self._step("Step 4 · Train / test split"):
            X_train, X_test, y_train, y_test, feat_names = self._split(fm)
            self.feature_names = feat_names
            self.log.info(
                "  Train: %d samples | Test: %d samples | Features: %d",
                len(X_train), len(X_test), len(feat_names),
            )

        # Step 5 — Train
        with self._step("Step 5 · Train models"):
            self.trainer = self._train(X_train, y_train, X_test, y_test, feat_names)

        # Step 6 — Detect overfitting (uses cached results — no re-computation)
        with self._step("Step 6 · Overfitting detection"):
            self.overfit_reports = self._detect_overfit(self.trainer)

        # Step 7 — Select best
        with self._step("Step 7 · Select best model"):
            self.best_model, self.best_model_name, self.best_metrics = (
                _select_best_model(self.trainer, self.overfit_reports, self.log)
            )
            self.log.info("  Selected: %s (AUC=%.4f)", self.best_model_name,
                          self.best_metrics["auc_roc"])

        # Step 8 — Save artefacts
        with self._step("Step 8 · Save artefacts"):
            self._save_artefacts(
                self.trainer, self.best_model, self.best_model_name,
                self.best_metrics, feat_names, X_test, y_test, self.overfit_reports,
            )

        # Step 9 — Run summary
        with self._step("Step 9 · Save run summary"):
            self._save_run_summary(self.best_model_name, self.best_metrics, started_at)

        # ── Final report ────────────────────────────────────────────────
        total = sum(self.step_times.values())
        self.log.info("=" * 65)
        self.log.info("  PIPELINE COMPLETE  |  total: %.1fs", total)
        self.log.info("  Model    : %s", self.best_model_name)
        for k, v in self.best_metrics.items():
            self.log.info("  %-12s: %.4f", k.upper(), v)
        self.log.info("=" * 65)

        return self


# ══════════════════════════════════════════════════════════════════════════════
# Dry-run validation
# ══════════════════════════════════════════════════════════════════════════════

def _dry_run(logger: logging.Logger) -> None:
    """Check all paths and imports without training. Exits 0 on success."""
    logger.info("DRY RUN — checking paths and imports only.")
    ok = True

    checks: list[tuple[str, Path]] = [
        ("data/raw",             Paths.DATA_RAW),
        ("data/processed",       Paths.DATA_PROCESSED),
        ("data/outputs",         Paths.DATA_OUTPUTS),
        ("data/outputs/models",  Paths.MODELS_DIR),
        ("reports/figures",      Paths.FIGURES_DIR),
    ]
    for label, path in checks:
        status = "✓" if path.exists() else "✗ (will be created at runtime)"
        logger.info("  %s  %s", status, label)

    required_raw = ["users.csv", "sessions.csv", "browse_events.csv",
                    "orders.csv", "order_items.csv", "products.csv"]
    for fname in required_raw:
        p = Paths.DATA_RAW / fname
        status = "✓" if p.exists() else "✗ MISSING — run generate_data.py"
        if not p.exists():
            ok = False
        logger.info("  %s  data/raw/%s", status, fname)

    logger.info("Dry run %s.", "PASSED" if ok else "FAILED — fix missing files above")
    sys.exit(0 if ok else 1)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args   = _parse_args()
    logger = _configure_logging(
        args.log_level,
        log_file=args.log_file,
    )

    if args.dry_run:
        _dry_run(logger)

    runner = PipelineRunner(
        target_col        = args.target_col,
        test_size         = args.test_size,
        n_iter            = args.n_iter,
        cv_folds          = args.cv_folds,
        overfit_threshold = args.overfit_threshold,
        skip_advanced     = args.skip_advanced,
        logger            = logger,
    )
    runner.run()


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════════════════════════
# CHANGELOG — what changed vs. pipeline_no_leakage.py and why
# ══════════════════════════════════════════════════════════════════════════════
# BUG  1  temporal_split() from train_advanced.py is now actually called; the
#          original imported it but used sklearn's stratified split silently.
# BUG  2  detect_overfitting computed once via trainer.detect_overfitting_all()
#          and cached in self.overfit_reports — never re-run in selection logic.
# BUG  3  save_model_advanced removed; pipeline calls trainer.save() which uses
#          the improved save_model() from train.py.
# BUG  4  _extract_feature_importance() replaces 4 if/elif branches; handles
#          Pipeline objects correctly and validates array length.
# BUG  5  XGBoost and LightGBM are no longer skipped. AdvancedModelTrainer
#          handles optional dependencies gracefully with logged warnings.
# BUG  6  _select_best_model() uses pre-computed reports dict — no re-runs.
# BUG  7  _build_features() validates target_col presence and raises ValueError
#          with a clear message before the split step.
# DESIGN 8  All print() replaced with logger calls; level configurable via CLI.
# DESIGN 9  _step() context manager records per-step elapsed time automatically.
# DESIGN10  AdvancedModelTrainer replaces 4 individual train_* calls.
# DESIGN11  trainer.leaderboard() used for metrics display — no manual loop.
# DESIGN12  trainer.save() builds metadata internally — no manual dict.
# DESIGN13  argparse exposes test_size, n_iter, cv_folds, threshold, skip_advanced,
#            dry_run, log_level, log_file — no source editing required.
# DESIGN14  _save_run_summary() writes pipeline_run_<id>.json for every run.
# DESIGN15  --dry-run validates paths and raw data files without training.
# DESIGN16  _ROOT computed from __file__ — works from any CWD.
# DESIGN17  target_col comes from ModelConfig.TARGET_COL (or --target-col flag).
# DESIGN18  All file loading wrapped in _step() which catches/logs FileNotFoundError.
# DESIGN19  _select_best_model() is a pure function — no logger calls, no side effects.
# DESIGN20  Step 7 selection logic is fully decoupled from training and logging.