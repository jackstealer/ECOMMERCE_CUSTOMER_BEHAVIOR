"""
Phase 2 — Data Cleaning & Preprocessing
========================================
Runs the full cleaning pipeline on all 6 raw tables,
creates the will_purchase target column, and saves
cleaned data to data/processed/ as Parquet files.

Run: python notebooks/02_data_cleaning.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pandas as pd
from config.settings import settings
from src.data.preprocess import clean_data
from src.data.validation import DataValidator


def main() -> dict:
    """Main function to run data cleaning pipeline."""

    print("=" * 80)
    print("PHASE 2: DATA CLEANING & PREPROCESSING")
    print("=" * 80)

    # ── Run cleaning pipeline ────────────────────────────────────
    print("\n1. Running cleaning pipeline ...")
    tables = clean_data(settings.RAW_DATA_DIR)

    print(f"\n   Tables cleaned: {list(tables.keys())}")
    print("\n   Row counts after cleaning:")
    for name, df in tables.items():
        print(f"   {name:<20} {len(df):>8,} rows  | {df.shape[1]} cols")

    # ── Target variable check ────────────────────────────────────
    if "users" in tables and "will_purchase" in tables["users"].columns:
        u = tables["users"]
        buyers     = u["will_purchase"].sum()
        non_buyers = len(u) - buyers
        print(f"\n2. Target variable — will_purchase:")
        print(f"   Buyers     (1): {buyers:,}  ({buyers/len(u)*100:.1f}%)")
        print(f"   Non-buyers (0): {non_buyers:,}  ({non_buyers/len(u)*100:.1f}%)")
    else:
        print("\n⚠️  'will_purchase' column not found in users table.")

    # ── Data quality validation ──────────────────────────────────
    print("\n3. Running data quality checks ...")
    validator = DataValidator()

    # Referential integrity checks
    if "sessions" in tables and "users" in tables:
        validator.check_referential_integrity(
            tables["sessions"], tables["users"],
            child_fk="user_id", parent_pk="user_id",
            child_table="sessions", parent_table="users",
        )
    if "orders" in tables and "users" in tables:
        validator.check_referential_integrity(
            tables["orders"], tables["users"],
            child_fk="user_id", parent_pk="user_id",
            child_table="orders", parent_table="users",
        )
    if "order_items" in tables and "orders" in tables:
        validator.check_referential_integrity(
            tables["order_items"], tables["orders"],
            child_fk="order_id", parent_pk="order_id",
            child_table="order_items", parent_table="orders",
        )
    if "browse_events" in tables and "products" in tables:
        validator.check_referential_integrity(
            tables["browse_events"], tables["products"],
            child_fk="product_id", parent_pk="product_id",
            child_table="browse_events", parent_table="products",
        )

    if validator.validation_errors:
        print(f"   ❌ Validation errors: {validator.validation_errors}")
    else:
        print("   ✅ All referential integrity checks passed")

    if validator.warnings:
        print(f"   ⚠️  Warnings: {validator.warnings}")

    # ── Save cleaned tables as Parquet ───────────────────────────
    print(f"\n4. Saving cleaned tables to {settings.PROCESSED_DATA_DIR}/")
    settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name, df in tables.items():
        # Save both Parquet (fast) and CSV (human-readable)
        df.to_parquet(settings.PROCESSED_DATA_DIR / f"{name}.parquet", index=False)
        df.to_csv(settings.PROCESSED_DATA_DIR / f"{name}.csv", index=False)
        print(f"   Saved {name}.parquet  ({len(df):,} rows)")

    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE")
    print("=" * 80)

    return tables


if __name__ == "__main__":
    main()
