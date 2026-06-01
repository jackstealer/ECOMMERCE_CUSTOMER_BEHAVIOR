"""
src/features/build_features.py
Reusable feature engineering functions — mirrors 04_feature_engineering.py

Key fixes vs original:
  - Fixed browse_events variable reference bug (was using original df instead of cleaned copy)
  - order_frequency outlier capped at 30 orders/month
  - create_rfm_features extracted to shared helper (no more dashboard duplication)
  - engineer_features saves will_purchase label onto feature matrix
"""

from pathlib import Path
import pandas as pd
import numpy as np


def create_session_features(
    sessions: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate sessions to user level."""
    sessions = sessions.copy()
    sessions["session_start"] = pd.to_datetime(sessions["session_start"])

    num_cols = [
        c for c in
        ["duration_secs_clean", "pages_visited_clean", "engagement_score", "bounced"]
        if c in sessions.columns
    ]
    agg = {col: ["mean", "max", "sum"] for col in num_cols}
    agg["session_id"] = "count"

    sess_agg = sessions.groupby("user_id").agg(agg)
    sess_agg.columns = ["_".join(c).strip("_") for c in sess_agg.columns]
    sess_agg = sess_agg.rename(columns={"session_id_count": "total_sessions"})

    last_sess = (
        sessions.groupby("user_id")["session_start"]
        .max()
        .rename("last_session_date")
        .reset_index()
    )
    last_sess["days_since_last_session"] = (
        snapshot - last_sess["last_session_date"]
    ).dt.days
    sess_agg = sess_agg.reset_index().merge(
        last_sess[["user_id", "days_since_last_session"]],
        on="user_id",
        how="left",
    )

    return sess_agg


def create_browse_features(
    browse_events: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate browse events to user level.

    FIX: was accidentally aggregating from original `browse_events` instead of
    the cleaned copy `be`, so column renames (e.g. time_spent_secs_clean) were
    never applied to the aggregated data.
    """
    # Always work on the passed-in DataFrame (already cleaned by preprocess.py)
    be = browse_events.copy()
    time_col = (
        "time_spent_secs_clean" if "time_spent_secs_clean" in be.columns
        else "time_spent_secs"
    )

    # FIX: aggregate from `be` (the copy), not the original parameter
    agg = be.groupby("user_id").agg(
        total_events            =("event_id",       "count"),
        unique_products_browsed =("product_id",     "nunique"),
        avg_time_per_event      =(time_col,          "mean"),
        total_cart_adds         =("added_to_cart",   "sum"),
        total_wishlists         =("wishlisted",      "sum"),
    ).reset_index()

    agg["cart_add_rate"] = (
        agg["total_cart_adds"] / agg["total_events"].replace(0, np.nan)
    ).fillna(0)
    agg["wishlist_rate"] = (
        agg["total_wishlists"] / agg["total_events"].replace(0, np.nan)
    ).fillna(0)
    agg["view_to_cart_ratio"] = agg["cart_add_rate"]  # alias used in dashboard

    cat_div = (
        be.merge(products[["product_id", "category"]], on="product_id", how="left")
        .groupby("user_id")["category"]
        .nunique()
        .reset_index()
        .rename(columns={"category": "unique_categories_browsed"})
    )
    agg = agg.merge(cat_div, on="user_id", how="left")
    return agg


def create_order_features(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate orders to user level."""
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    order_agg = orders.groupby("user_id").agg(
        total_orders     =("order_id",     "count"),
        total_spend      =("total_amount", "sum"),
        avg_order_value  =("total_amount", "mean"),
        max_order_value  =("total_amount", "max"),
        avg_discount_pct =("discount_pct", "mean"),
        last_order_date  =("order_date",   "max"),
        first_order_date =("order_date",   "min"),
    ).reset_index()

    order_agg["days_since_last_order"] = (
        snapshot - order_agg["last_order_date"]
    ).dt.days
    order_agg["customer_lifespan_days"] = (
        order_agg["last_order_date"] - order_agg["first_order_date"]
    ).dt.days.clip(lower=1)

    order_agg["order_frequency"] = (
        order_agg["total_orders"]
        / (order_agg["customer_lifespan_days"] / 30).replace(0, np.nan)
    ).fillna(order_agg["total_orders"])
    # Cap extreme outlier values (e.g. 1-day customers with 5 orders → 150/month)
    order_agg["order_frequency"] = order_agg["order_frequency"].clip(upper=30)

    order_agg = order_agg.drop(columns=["last_order_date", "first_order_date"])
    return order_agg


def create_rfm_features(
    orders: pd.DataFrame,
    snapshot: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute RFM scores and segment labels for each buyer.

    This is the single authoritative implementation — used by both the
    feature pipeline and the Streamlit dashboard (no more duplication).
    """
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    if snapshot is None:
        snapshot = orders["order_date"].max() + pd.Timedelta(days=1)

    rfm = orders.groupby("user_id").agg(
        recency  =("order_date",   lambda x: (snapshot - x.max()).days),
        frequency=("order_id",     "count"),
        monetary =("total_amount", "sum"),
    ).reset_index()

    rfm["R_score"] = pd.qcut(rfm["recency"],   q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(
        rfm["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]
    ).astype(int)
    rfm["M_score"]   = pd.qcut(rfm["monetary"], q=4, labels=[1, 2, 3, 4]).astype(int)
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    def _segment(row) -> str:
        if   row["RFM_score"] >= 10:                         return "Champions"
        elif row["R_score"]   >= 3 and row["F_score"] >= 3: return "Loyal"
        elif row["R_score"]   >= 3:                          return "Recent"
        elif row["F_score"]   >= 3:                          return "At Risk"
        elif row["RFM_score"] <= 5:                          return "Lost"
        else:                                                 return "Potential"

    rfm["segment"] = rfm.apply(_segment, axis=1)
    return rfm


def engineer_features(
    tables: dict,
    snapshot: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline — returns the master feature matrix.

    The returned DataFrame includes:
      - All user-level features merged from sessions, browse events, and orders
      - 'will_purchase' target column (0/1)

    Usage:
        from src.features.build_features import engineer_features
        feature_matrix = engineer_features(tables)
    """
    if snapshot is None:
        snapshot = (
            tables["orders"]["order_date"].max()
            if "orders" in tables
            else pd.Timestamp.now()
        )
    snapshot = pd.Timestamp(snapshot)

    # User base
    always = ["user_id", "age", "account_age_days", "will_purchase"]
    enc_cols = [
        c for c in tables["users"].columns
        if c in ("membership_encoded", "is_premium")
        or c.startswith(("gender_", "device_"))
    ]
    # Keep will_purchase only if it exists
    keep = [c for c in always if c in tables["users"].columns] + enc_cols
    user_f = tables["users"][keep].copy()
    user_f["is_new_user"]     = (user_f["account_age_days"] < 30).astype(int)
    user_f["is_veteran_user"] = (user_f["account_age_days"] > 365).astype(int)

    sess_f   = create_session_features(tables["sessions"],    snapshot)
    browse_f = create_browse_features(tables["browse_events"], tables["products"])
    order_f  = create_order_features(tables["orders"], tables["order_items"], snapshot)

    fm = (
        user_f
        .merge(sess_f,   on="user_id", how="left")
        .merge(browse_f, on="user_id", how="left")
        .merge(order_f,  on="user_id", how="left")
    )

    # Fill numeric NAs for users with no sessions / events / orders
    num_cols = [
        c for c in fm.columns
        if c not in ("user_id", "will_purchase")
        and fm[c].dtype in ["float64", "int64"]
    ]
    fm[num_cols] = fm[num_cols].fillna(0)

    return fm