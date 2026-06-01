"""
Application configuration with environment variable support.
All magic constants for data generation live here so they can be
changed in one place without touching source code.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env (if present)
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # ── Project paths ────────────────────────────────────────────
    BASE_DIR          = Path(__file__).resolve().parent.parent
    DATA_DIR          = BASE_DIR / "data"
    RAW_DATA_DIR      = DATA_DIR / "raw"
    PROCESSED_DATA_DIR= DATA_DIR / "processed"
    OUTPUT_DIR        = DATA_DIR / "outputs"
    MODEL_DIR         = OUTPUT_DIR / "models"
    LOGS_DIR          = BASE_DIR / "logs"
    FIGURES_DIR       = BASE_DIR / "reports" / "figures"

    # ── Database ─────────────────────────────────────────────────
    DB_HOST:     str = os.getenv("DB_HOST", "localhost")
    DB_PORT:     int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME:     str = os.getenv("DB_NAME", "ecommerce_db")
    DB_USER:     str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL from parts (evaluated at call time, not class definition)."""
        if self.DB_USER and self.DB_PASSWORD:
            return (
                f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
                f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            )
        return ""

    # ── API ──────────────────────────────────────────────────────
    API_BASE_URL: str = os.getenv("API_BASE_URL", "")
    API_KEY:      str = os.getenv("API_KEY", "")
    API_TIMEOUT:  int = int(os.getenv("API_TIMEOUT", "30"))

    # ── AWS ──────────────────────────────────────────────────────
    AWS_ACCESS_KEY_ID:     Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION:            str           = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET:             Optional[str] = os.getenv("S3_BUCKET")

    # ── Model ─────────────────────────────────────────────────────
    # MODEL_PATH evaluated lazily via property so MODEL_DIR is already resolved
    MODEL_VERSION:         str   = os.getenv("MODEL_VERSION", "v1.0.0")
    PREDICTION_THRESHOLD:  float = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))

    @property
    def MODEL_PATH(self) -> str:
        return os.getenv("MODEL_PATH", str(self.MODEL_DIR))

    # ── Monitoring & Logging ─────────────────────────────────────
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    LOG_LEVEL:  str           = os.getenv("LOG_LEVEL", "INFO")

    @property
    def LOG_FILE(self) -> str:
        return os.getenv("LOG_FILE", str(self.LOGS_DIR / "app.log"))

    # ── Feature Flags ────────────────────────────────────────────
    ENABLE_REAL_TIME_PREDICTIONS: bool = (
        os.getenv("ENABLE_REAL_TIME_PREDICTIONS", "true").lower() == "true"
    )
    ENABLE_AB_TESTING: bool = (
        os.getenv("ENABLE_AB_TESTING", "false").lower() == "true"
    )
    ENABLE_CACHING: bool = (
        os.getenv("ENABLE_CACHING", "true").lower() == "true"
    )

    # ── Application ───────────────────────────────────────────────
    ENVIRONMENT: str  = os.getenv("ENVIRONMENT", "development")
    DEBUG:       bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── Data Generation Config (centralised magic constants) ──────
    DATA_CONFIG = {
        # Date range for synthetic data
        "start_date": "2024-06-01",
        "end_date":   "2025-05-31",
        # Table sizes
        "n_users":    5_000,
        "n_products":   500,
        "n_sessions": 20_000,
        "n_orders":    8_000,
        # User demographic distributions
        "gender_probs":     [0.47, 0.47, 0.06],   # Male, Female, Other
        "device_probs":     [0.55, 0.35, 0.10],   # mobile, desktop, tablet
        "membership_probs": [0.55, 0.25, 0.15, 0.05],  # free, silver, gold, platinum
        # Session behaviour
        "bounce_rate":      0.38,
        "avg_session_secs": 600,
        # Referral source distribution (realistic channel mix)
        "referral_probs":   [0.30, 0.20, 0.20, 0.15, 0.10, 0.05],
        # Order value range (USD)
        "order_value_min":  5.0,
        "order_value_max":  1500.0,
        # Browse event type probabilities
        "event_probs": [0.50, 0.25, 0.14, 0.06, 0.05],  # view,click,cart,remove,wishlist
    }

    def __init__(self):
        """Create necessary directories on startup."""
        for d in [self.LOGS_DIR, self.MODEL_DIR, self.RAW_DATA_DIR,
                  self.PROCESSED_DATA_DIR, self.OUTPUT_DIR, self.FIGURES_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# Global settings instance — import this everywhere
settings = Settings()
