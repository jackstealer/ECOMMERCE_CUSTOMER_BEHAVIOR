"""
src/data/preprocess.py
Reusable cleaning functions — mirrors the logic in 02_data_cleaning.py
so the pipeline can be run as a script: python src/data/preprocess.py

Key changes vs original:
  - Fixed age bin labels to match pd.cut edges correctly
  - Added will_purchase target column (user bought at least once)
  - Corrected browse_events variable reference bug
  - clean_data() uses settings.RAW_DATA_DIR / PROCESSED_DATA_DIR
"""

from pathlib import Path

import pandas as pd
import numpy as np


# ── Loaders ───────────────────────────────────────────────────

def load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every CSV from data/raw/ and return as {name: DataFrame}."""
    return {f.stem: pd.read_csv(f) for f in sorted(data_dir.glob("*.csv"))}


# ── Cleaning helpers ──────────────────────────────────────────

def parse_datetimes(tables: dict) -> dict:
    """Convert string timestamps to datetime in sessions, orders, users."""
    if "sessions" in tables:
        for col in ["session_start", "session_end"]:
            if col in tables["sessions"].columns:
                tables["sessions"][col] = pd.to_datetime(tables["sessions"][col])

    if "orders" in tables:
        tables["orders"]["order_date"] = pd.to_datetime(tables["orders"]["order_date"])

    if "users" in tables:
        tables["users"]["signup_date"] = pd.to_datetime(tables["users"]["signup_date"])

    return tables


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing values.
    - Numeric columns  → median (default) or mean
    - Object columns   → mode
    """
    df = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if df[col].isnull().any():
            fill = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill)
    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_duplicates(df: pd.DataFrame, pk_col: str | None = None) -> pd.DataFrame:
    """Drop fully-duplicate rows. If pk_col given, also deduplicate on that key."""
    df = df.drop_duplicates()
    if pk_col and pk_col in df.columns:
        df = df.drop_duplicates(subset=[pk_col], keep="first")
    return df.reset_index(drop=True)


def clip_outliers(df: pd.DataFrame, col: str, percentile: float = 0.95) -> pd.DataFrame:
    """Clip values in col at the given upper percentile (IQR-safe, no hardcodes)."""
    df = df.copy()
    upper = df[col].quantile(percentile)
    df[col] = df[col].clip(upper=upper)
    return df


def log_transform(df: pd.DataFrame, col: str, new_col: str | None = None) -> pd.DataFrame:
    """Apply log1p transform. Stores result in new_col (defaults to col + '_log')."""
    df  = df.copy()
    out = new_col or f"{col}_log"
    df[out] = np.log1p(df[col])
    return df


def encode_ordinal(df: pd.DataFrame, col: str, order: list) -> pd.DataFrame:
    """Map col to integer ranks defined by order list (0 = first item)."""
    df = df.copy()
    mapping = {v: i for i, v in enumerate(order)}
    df[f"{col}_encoded"] = df[col].map(mapping)
    return df


# ── Full cleaning pipeline ─────────────────────────────────────

def clean_data(raw_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """
    Full cleaning pipeline.
    Returns dict of cleaned DataFrames keyed by table name.

    Usage:
        from src.data.preprocess import clean_data
        tables = clean_data()

    Also adds the ML target column:
        will_purchase: 1 if the user has at least one order, 0 otherwise
    """
    # ── Resolve path ──────────────────────────────────────────
    if raw_dir is None:
        try:
            from config.settings import settings
            raw_dir = settings.RAW_DATA_DIR
        except ImportError:
            _cwd = Path(__file__).resolve().parent
            for candidate in [
                _cwd.parent.parent / "data" / "raw",
                Path().resolve() / "data" / "raw",
            ]:
                if candidate.exists():
                    raw_dir = candidate
                    break

    if raw_dir is None or not raw_dir.exists():
        raise FileNotFoundError(
            "data/raw/ not found — run 'python src/data/generate_data.py' first."
        )

    tables = load_raw(raw_dir)
    tables = parse_datetimes(tables)

    # ── Sessions ──────────────────────────────────────────────
    if "sessions" in tables:
        s = tables["sessions"]
        s = clip_outliers(s, "duration_secs",  0.95)
        s = clip_outliers(s, "pages_visited",  0.99)
        # Rename clipped cols so downstream code can find them
        s = s.rename(columns={
            "duration_secs": "duration_secs_clean",
            "pages_visited": "pages_visited_clean",
        })
        s["bounced"] = s["bounced"].astype(int)
        # Derived feature used in feature engineering
        s["engagement_score"] = (
            (s["pages_visited_clean"] / s["pages_visited_clean"].max()) * 0.5
            + (1 - s["bounced"]) * 0.5
        ).round(4)
        tables["sessions"] = s

    # ── Orders ────────────────────────────────────────────────
    if "orders" in tables:
        o = tables["orders"]
        o = clip_outliers(o, "total_amount", 0.95)
        o = log_transform(o, "total_amount", "log_total_amount")
        o = encode_ordinal(
            o, "status",
            ["cancelled", "returned", "processing", "shipped", "delivered"],
        )
        tables["orders"] = o

    # ── Users ─────────────────────────────────────────────────
    if "users" in tables:
        u = tables["users"]
        u = encode_ordinal(u, "membership", ["free", "silver", "gold", "platinum"])
        u["is_premium"] = u["membership"].isin(["gold", "platinum"]).astype(int)

        # FIX: Correct age bins that match the labels
        # bins=[18,25,35,50,70] → (18–25], (25–35], (35–50], (50–70]
        age_bins   = [17, 25, 35, 50, 71]   # right=True (default), so 18–25, 26–35, 36–50, 51+
        age_labels = ["18-25", "26-35", "36-50", "51+"]
        u["age_group"] = pd.cut(u["age"], bins=age_bins, labels=age_labels)

        ref = (
            tables["orders"]["order_date"].max()
            if "orders" in tables
            else pd.Timestamp.now()
        )
        u["account_age_days"] = (ref - u["signup_date"]).dt.days.clip(lower=0)

        # ── ML TARGET: will_purchase ──────────────────────────
        # 1 if the user has at least one order in the dataset, else 0
        if "orders" in tables:
            buyer_ids = set(tables["orders"]["user_id"].unique())
            u["will_purchase"] = u["user_id"].isin(buyer_ids).astype(int)
        else:
            u["will_purchase"] = 0

        tables["users"] = u

    # ── Browse events ─────────────────────────────────────────
    if "browse_events" in tables:
        b = tables["browse_events"].copy()   # work on a copy consistently
        b = clip_outliers(b, "time_spent_secs", 0.95)
        b = b.rename(columns={"time_spent_secs": "time_spent_secs_clean"})
        b["added_to_cart"] = (b["event_type"] == "add_to_cart").astype(int)
        b["wishlisted"]    = (b["event_type"] == "add_to_wishlist").astype(int)
        tables["browse_events"] = b

    return tables


# ── CLI entry-point ───────────────────────────────────────────

if __name__ == "__main__":
    tables = clean_data()
    for name, df in tables.items():
        print(f"  {name:<20} {len(df):>8,} rows  |  {df.shape[1]} cols")