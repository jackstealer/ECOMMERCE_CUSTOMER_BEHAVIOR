"""
Phase 3 — Exploratory Data Analysis
======================================
Performs EDA across all 6 cleaned tables:
  - User demographics & membership breakdown
  - Session behaviour (referral, bounce, duration)
  - Browse funnel (event types, top products)
  - Order patterns & revenue
  - RFM segmentation (using shared helper, not duplicated code)
  - will_purchase target analysis

Saves plots to reports/figures/

Run: python notebooks/03_eda.py
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
import seaborn as sns

from config.settings import settings
from src.features.build_features import create_rfm_features   # single authoritative RFM

sns.set_style("whitegrid")
FIGS = settings.FIGURES_DIR


def load_processed() -> dict:
    proc = settings.PROCESSED_DATA_DIR
    tables = {}
    for p in sorted(proc.glob("*.parquet")):
        tables[p.stem] = pd.read_parquet(p)
    # fallback to CSV
    for p in sorted(proc.glob("*.csv")):
        if p.stem not in tables:
            tables[p.stem] = pd.read_csv(p)
    return tables


def main() -> dict:
    """Main function to run EDA."""

    print("=" * 80)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    tables = load_processed()
    if not tables:
        print("❌  No processed data found — run Phase 2 first.")
        return {}

    FIGS.mkdir(parents=True, exist_ok=True)

    users         = tables.get("users")
    sessions      = tables.get("sessions")
    orders        = tables.get("orders")
    browse_events = tables.get("browse_events")
    products      = tables.get("products")

    # ── 1. Target variable distribution ─────────────────────────
    if users is not None and "will_purchase" in users.columns:
        print("\n1. Target variable (will_purchase):")
        vc = users["will_purchase"].value_counts().sort_index()
        for label, count in vc.items():
            print(f"   {'Buyer' if label else 'Non-buyer':10}  {count:>6,}  "
                  f"({count/len(users)*100:.1f}%)")

    # ── 2. User demographics ─────────────────────────────────────
    if users is not None:
        print("\n2. User demographics ...")
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("User Demographics", fontsize=13, fontweight="bold")

        # Age histogram
        axes[0, 0].hist(users["age"], bins=25, edgecolor="white",
                        color="#4f8ef7", alpha=0.85)
        axes[0, 0].axvline(users["age"].mean(), color="red", linestyle="--",
                           label=f"Mean: {users['age'].mean():.0f}")
        axes[0, 0].set_title("Age Distribution")
        axes[0, 0].set_xlabel("Age"); axes[0, 0].set_ylabel("Count")
        axes[0, 0].legend()

        # Membership
        mem_vc = users["membership"].value_counts().reindex(
            ["free", "silver", "gold", "platinum"]
        )
        axes[0, 1].bar(mem_vc.index, mem_vc.values,
                       color=["#aaa", "#c0c0c0", "#ffd700", "#b9f2ff"], alpha=0.9)
        axes[0, 1].set_title("Membership Tier"); axes[0, 1].set_ylabel("Users")

        # Gender
        gen_vc = users["gender"].value_counts()
        axes[0, 2].bar(gen_vc.index, gen_vc.values, color=["#4f8ef7","#fc8d62","#8da0cb"])
        axes[0, 2].set_title("Gender Distribution"); axes[0, 2].set_ylabel("Users")

        # Device type
        dev_vc = users["device_type"].value_counts()
        axes[1, 0].pie(dev_vc.values, labels=dev_vc.index, autopct="%1.1f%%",
                       startangle=140)
        axes[1, 0].set_title("Device Type")

        # Age group
        if "age_group" in users.columns:
            ag_vc = users["age_group"].value_counts().sort_index()
            axes[1, 1].bar(ag_vc.index.astype(str), ag_vc.values,
                           color="#66c2a5", alpha=0.9)
            axes[1, 1].set_title("Age Groups"); axes[1, 1].set_ylabel("Users")

        # Purchase rate by membership
        if "will_purchase" in users.columns:
            pr = users.groupby("membership")["will_purchase"].mean().reindex(
                ["free", "silver", "gold", "platinum"]
            ) * 100
            axes[1, 2].bar(pr.index, pr.values,
                           color=["#aaa", "#c0c0c0", "#ffd700", "#b9f2ff"])
            axes[1, 2].set_title("Purchase Rate by Membership (%)")
            axes[1, 2].set_ylabel("Purchase Rate (%)")

        plt.tight_layout()
        _save(fig, "03_user_demographics.png")

    # ── 3. Session behaviour ─────────────────────────────────────
    if sessions is not None:
        print("3. Session behaviour ...")
        sessions["session_start"] = pd.to_datetime(sessions["session_start"])

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Session Behaviour", fontsize=13, fontweight="bold")

        # Sessions per month
        sessions.groupby(sessions["session_start"].dt.to_period("M")).size() \
            .plot(kind="line", marker="o", ax=axes[0, 0], color="#4f8ef7")
        axes[0, 0].set_title("Sessions per Month")
        axes[0, 0].tick_params(axis="x", rotation=30)

        # Referral source breakdown (now weighted, not uniform random)
        ref_vc = sessions["referral_source"].value_counts()
        axes[0, 1].barh(ref_vc.index, ref_vc.values, color="#fc8d62", alpha=0.85)
        axes[0, 1].set_title("Sessions by Referral Source")

        # Duration distribution (use clean col if available)
        dur_col = "duration_secs_clean" if "duration_secs_clean" in sessions else "duration_secs"
        axes[1, 0].hist(sessions[dur_col] / 60, bins=40,
                        edgecolor="white", color="#a6d854", alpha=0.85)
        axes[1, 0].set_title("Session Duration (minutes)")

        # Bounce rate
        bounce = sessions["bounced"].mean() * 100
        axes[1, 1].bar(["Bounced", "Engaged"], [bounce, 100 - bounce],
                       color=["#e74c3c", "#2ecc71"], alpha=0.85)
        axes[1, 1].set_title(f"Bounce Rate: {bounce:.1f}%")
        axes[1, 1].set_ylabel("Percentage")

        plt.tight_layout()
        _save(fig, "03_session_behaviour.png")

    # ── 4. Browse funnel & product analysis ──────────────────────
    if browse_events is not None and products is not None:
        print("4. Browse funnel & product analysis ...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Browse Funnel", fontsize=13, fontweight="bold")

        # Event type distribution
        et_vc = browse_events["event_type"].value_counts()
        axes[0].barh(et_vc.index, et_vc.values, color="#8da0cb", alpha=0.85)
        axes[0].set_title("Browse Events by Type")

        # Sales by category
        cat_browse = (
            browse_events.merge(products[["product_id", "category"]],
                                on="product_id", how="left")
            .groupby("category").size()
            .sort_values()
        )
        axes[1].barh(cat_browse.index, cat_browse.values,
                     color="#66c2a5", alpha=0.85)
        axes[1].set_title("Browse Events by Category")

        plt.tight_layout()
        _save(fig, "03_browse_funnel.png")

    # ── 5. Order & revenue analysis ──────────────────────────────
    if orders is not None:
        print("5. Order & revenue analysis ...")
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Order Analysis", fontsize=13, fontweight="bold")

        axes[0].hist(orders["total_amount"], bins=40,
                     edgecolor="white", color="#4f8ef7", alpha=0.85)
        axes[0].set_title("Order Value Distribution ($)")

        status_vc = orders["status"].value_counts()
        axes[1].bar(status_vc.index, status_vc.values, color="#fc8d62", alpha=0.85)
        axes[1].set_title("Order Status"); axes[1].tick_params(axis="x", rotation=30)

        pay_vc = orders["payment_method"].value_counts()
        axes[2].pie(pay_vc.values, labels=pay_vc.index, autopct="%1.1f%%",
                    startangle=140)
        axes[2].set_title("Payment Methods")

        plt.tight_layout()
        _save(fig, "03_order_analysis.png")

    # ── 6. RFM segmentation ──────────────────────────────────────
    if orders is not None:
        print("6. RFM segmentation (using shared helper) ...")
        rfm = create_rfm_features(orders)  # uses the single authoritative function

        seg_counts = rfm["segment"].value_counts()
        seg_spend  = rfm.groupby("segment")["monetary"].mean().sort_values()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("RFM Customer Segments", fontsize=13, fontweight="bold")

        seg_counts.plot(kind="bar", ax=axes[0],
                        color=sns.color_palette("Set2", len(seg_counts)), alpha=0.85)
        axes[0].set_title("Customers per Segment")
        axes[0].tick_params(axis="x", rotation=20)

        seg_spend.plot(kind="barh", ax=axes[1], color="#a6d854", alpha=0.85)
        axes[1].set_title("Avg Spend per Segment ($)")

        plt.tight_layout()
        _save(fig, "03_rfm_segments.png")

        print("\n   RFM Segment Summary:")
        summary = rfm.groupby("segment").agg(
            count=("user_id", "count"),
            avg_spend=("monetary", "mean"),
            avg_orders=("frequency", "mean"),
            avg_recency=("recency", "mean"),
        ).round(1).sort_values("avg_spend", ascending=False)
        print(summary.to_string())

    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE")
    print("=" * 80)

    return tables


def _save(fig: plt.Figure, name: str):
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved → {path}")


if __name__ == "__main__":
    main()
