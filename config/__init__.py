"""Configuration module - exports settings and configuration classes."""

from pathlib import Path
from .settings import settings

# ── Path Configuration ────────────────────────────────────────────────
class Paths:
    """Centralized path configuration for the project."""
    BASE_DIR           = settings.BASE_DIR
    DATA_DIR           = settings.DATA_DIR
    DATA_RAW           = settings.RAW_DATA_DIR
    DATA_PROCESSED     = settings.PROCESSED_DATA_DIR
    DATA_OUTPUTS       = settings.OUTPUT_DIR
    MODELS_DIR         = settings.MODEL_DIR
    FIGURES_DIR        = settings.FIGURES_DIR
    LOGS_DIR           = settings.LOGS_DIR


# ── Model Configuration ────────────────────────────────────────────────
class ModelConfig:
    """ML model training configuration."""
    RANDOM_STATE      = 42
    TEST_SIZE         = 0.2
    CV_FOLDS          = 5
    N_ITER_TUNE       = 20
    TARGET_COL        = "will_purchase"
    PREDICTION_THRESHOLD = settings.PREDICTION_THRESHOLD


# ── Business Configuration ──────────────────────────────────────────────
class BusinessConfig:
    """Business logic and domain configuration."""
    # Customer segmentation thresholds
    HIGH_VALUE_THRESHOLD     = 500.0  # Total spend for high-value customers
    AT_RISK_DAYS_THRESHOLD   = 90     # Days inactive to be "at risk"
    NEW_CUSTOMER_DAYS        = 30     # Days since signup to be "new"
    VETERAN_CUSTOMER_DAYS    = 365    # Days active to be "veteran"
    
    # Purchase prediction adjustments
    CHURN_DECAY_START_DAYS   = 90     # Start decay after this many inactive days
    CHURN_DECAY_FACTOR       = 0.7    # Multiply probability by this for churning customers
    
    # Membership tiers (for encoding)
    MEMBERSHIP_TIERS = {
        "free": 0,
        "silver": 1,
        "gold": 2,
        "platinum": 3
    }
    
    # Device types (for encoding)
    DEVICE_TYPES = {
        "mobile": 0,
        "tablet": 1,
        "desktop": 2
    }


# Export all configuration classes and settings instance
__all__ = [
    "settings",
    "Paths",
    "ModelConfig", 
    "BusinessConfig"
]
