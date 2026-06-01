"""
Phase 4 — Feature Engineering
================================
Builds the master feature matrix from the 6 cleaned tables,
merging session, browse, and order aggregates onto the user table.

Output: data/processed/feature_matrix.parquet
         data/processed/feature_matrix.csv   (human-readable copy)

Run: python notebooks/04_feature_engineering.py
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import settings
from src.features.build_features import engineer_features, create_rfm_features


def load_processed() -> dict:
    proc = settings.PROCESSED_DATA_DIR
    tables = {}
    for p in sorted(proc.glob("*.parquet")):
        tables[p.stem] = pd.read_parquet(p)
    for p in sorted(proc.glob("*.csv")):
        if p.stem not in tables:
            tables[p.stem] = pd.read_csv(p)
    return tables


def main() -> pd.DataFrame:
    """Main function to run feature engineering pipeline."""

    print("=" * 80)
    print("PHASE 4: FEATURE ENGINEERING")
    print("=" * 80)

    tables = load_processed()
    if not tables:
        print("❌  No processed data found — run Phase 2 first.")
        return None

    # Convert datetime cols (Parquet preserves types; CSV does not)
    for col in ["session_start", "session_end"]:
        if "sessions" in tables and col in tables["sessions"].columns:
            tables["sessions"][col] = pd.to_datetime(tables["sessions"][col])
    if "orders" in tables:
        tables["orders"]["order_date"] = pd.to_datetime(tables["orders"]["order_date"])

    # ── Build feature matrix ─────────────────────────────────────
    print("\n1. Building feature matrix ...")
    fm = engineer_features(tables)

    print(f"\n   Feature matrix shape : {fm.shape}")
    print(f"   Features created     : {fm.shape[1] - 1}  (+ user_id)")

    if "will_purchase" in fm.columns:
        buyers = fm["will_purchase"].sum()
        print(f"\n2. Target variable (will_purchase):")
        print(f"   Buyers     (1): {buyers:,}  ({buyers/len(fm)*100:.1f}%)")
        print(f"   Non-buyers (0): {len(fm)-buyers:,}  "
              f"({(len(fm)-buyers)/len(fm)*100:.1f}%)")
    else:
        print("\n⚠️  'will_purchase' column not in feature matrix!")

    # ── Sample of new features ───────────────────────────────────
    print("\n3. Feature matrix columns:")
    cols = [c for c in fm.columns if c != "user_id"]
    for i, col in enumerate(cols, 1):
        print(f"   {i:>2}. {col}")

    # ── Feature correlation with target ─────────────────────────
    if "will_purchase" in fm.columns:
        print("\n4. Top features correlated with will_purchase:")
        num_cols = fm.select_dtypes(include=[np.number]).columns.tolist()
        corr = fm[num_cols].corrwith(fm["will_purchase"]).drop("will_purchase") \
                           .abs().sort_values(ascending=False)
        for feat, val in corr.head(10).items():
            print(f"   {feat:<40}  {val:.4f}")

        # Plot feature importances preview
        settings.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        corr.head(15).sort_values().plot(kind="barh", ax=ax, color="#4f8ef7", alpha=0.85)
        ax.set_title("Top Features Correlated with will_purchase (|Pearson r|)")
        ax.set_xlabel("|Correlation|")
        plt.tight_layout()
        path = settings.FIGURES_DIR / "04_feature_correlations.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n   Saved → {path}")

    # ── RFM segments (bonus insight) ────────────────────────────
    if "orders" in tables:
        print("\n5. RFM Customer Segmentation:")
        rfm = create_rfm_features(tables["orders"])
        seg_summary = rfm["segment"].value_counts()
        for seg, cnt in seg_summary.items():
            print(f"   {seg:<15}  {cnt:>5,} customers")

    # ── Save feature matrix ──────────────────────────────────────
    print(f"\n6. Saving feature matrix to {settings.PROCESSED_DATA_DIR}/")
    out_parquet = settings.PROCESSED_DATA_DIR / "feature_matrix.parquet"
    out_csv     = settings.PROCESSED_DATA_DIR / "feature_matrix.csv"
    fm.to_parquet(out_parquet, index=False)
    fm.to_csv(out_csv, index=False)
    print(f"   Saved feature_matrix.parquet  ({len(fm):,} rows × {fm.shape[1]} cols)")

    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE")
    print("=" * 80)

    return fm


if __name__ == "__main__":
    main()
