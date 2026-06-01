"""
Phase 1 — Data Collection & Understanding
==========================================
Loads all 6 raw CSV tables produced by generate_data.py,
explores structure, data types, missing values, and distributions.

Run: python notebooks/01_data_understanding.py
Or:  python pipeline.py  (runs everything in order)
"""

import sys
from pathlib import Path

# Allow running from notebooks/ OR from project root
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import settings

sns.set_style("whitegrid")


def main() -> dict:
    """Main function to run data understanding analysis."""

    print("=" * 80)
    print("PHASE 1: DATA COLLECTION & UNDERSTANDING")
    print("=" * 80)

    raw_dir = settings.RAW_DATA_DIR
    if not raw_dir.exists():
        print(f"\n❌  data/raw/ not found at {raw_dir}")
        print("   Run 'python src/data/generate_data.py' first.")
        return {}

    # ── Load all 6 tables ────────────────────────────────────────
    print("\n1. Loading raw data tables ...")
    tables = {}
    for csv_file in sorted(raw_dir.glob("*.csv")):
        tables[csv_file.stem] = pd.read_csv(csv_file)
        print(f"   {csv_file.name:<25}  {len(tables[csv_file.stem]):>8,} rows  "
              f"| {tables[csv_file.stem].shape[1]} cols")

    if not tables:
        print("   No CSV files found — did generate_data.py run?")
        return tables

    # ── Schema overview ──────────────────────────────────────────
    print("\n2. Schema overview:")
    for name, df in tables.items():
        print(f"\n  [{name}]")
        print(f"    Columns    : {list(df.columns)}")
        print(f"    Dtypes     : {df.dtypes.to_dict()}")
        print(f"    Missing    : {df.isnull().sum().sum()} cells")
        print(f"    Duplicates : {df.duplicated().sum()} rows")

    # ── Target variable: will_purchase ───────────────────────────
    if "orders" in tables and "users" in tables:
        buyer_ids  = set(tables["orders"]["user_id"].unique())
        n_buyers   = len(buyer_ids)
        n_users    = len(tables["users"])
        buyer_rate = n_buyers / n_users * 100
        print(f"\n3. Target variable (will_purchase):")
        print(f"   Users who placed ≥1 order : {n_buyers:,}  ({buyer_rate:.1f}%)")
        print(f"   Non-buyers                : {n_users - n_buyers:,}  "
              f"({100 - buyer_rate:.1f}%)")

    # ── Key distributions visualisation ─────────────────────────
    print("\n4. Generating visualisations ...")
    settings.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Phase 1 — Data Overview", fontsize=14, fontweight="bold")

    # Users: age distribution
    if "users" in tables:
        u = tables["users"]
        axes[0, 0].hist(u["age"], bins=25, edgecolor="white", color="#4f8ef7", alpha=0.85)
        axes[0, 0].axvline(u["age"].mean(), color="red", linestyle="--",
                           label=f"Mean: {u['age'].mean():.0f}")
        axes[0, 0].set_title("User Age Distribution")
        axes[0, 0].set_xlabel("Age"); axes[0, 0].set_ylabel("Count")
        axes[0, 0].legend()

        # Membership
        mem_vc = u["membership"].value_counts().reindex(
            ["free", "silver", "gold", "platinum"]
        )
        axes[0, 1].bar(mem_vc.index, mem_vc.values, color=["#aaa", "#c0c0c0", "#ffd700", "#b9f2ff"])
        axes[0, 1].set_title("Membership Tier Breakdown")
        axes[0, 1].set_xlabel("Tier"); axes[0, 1].set_ylabel("Users")

    # Orders: value distribution
    if "orders" in tables:
        o = tables["orders"]
        axes[0, 2].hist(o["total_amount"], bins=40, edgecolor="white",
                        color="#2ecc71", alpha=0.85)
        axes[0, 2].set_title("Order Value Distribution")
        axes[0, 2].set_xlabel("Amount ($)"); axes[0, 2].set_ylabel("Count")

    # Sessions: duration
    if "sessions" in tables:
        s = tables["sessions"]
        axes[1, 0].hist(s["duration_secs"] / 60, bins=40, edgecolor="white",
                        color="#e67e22", alpha=0.85)
        axes[1, 0].set_title("Session Duration (minutes)")
        axes[1, 0].set_xlabel("Minutes"); axes[1, 0].set_ylabel("Count")

    # Browse events: type breakdown
    if "browse_events" in tables:
        be = tables["browse_events"]
        et_vc = be["event_type"].value_counts()
        axes[1, 1].barh(et_vc.index, et_vc.values, color="#8e44ad", alpha=0.85)
        axes[1, 1].set_title("Browse Event Types")
        axes[1, 1].set_xlabel("Count")

    # Products: category split
    if "products" in tables:
        p = tables["products"]
        cat_vc = p["category"].value_counts()
        axes[1, 2].pie(cat_vc.values, labels=cat_vc.index, autopct="%1.0f%%",
                       startangle=140)
        axes[1, 2].set_title("Product Categories")

    plt.tight_layout()
    out_path = settings.FIGURES_DIR / "01_data_understanding.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {out_path}")

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)

    return tables


if __name__ == "__main__":
    main()
