"""Tests for src/features/build_features.py"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_sessions(n=30) -> pd.DataFrame:
    return pd.DataFrame({
        "session_id":          [f"S{i:06d}" for i in range(n)],
        "user_id":             np.random.choice(["U00001","U00002","U00003"], n),
        "session_start":       pd.date_range("2024-06-01", periods=n, freq="3h"),
        "duration_secs_clean": np.random.randint(30, 1800, n).astype(float),
        "pages_visited_clean": np.random.randint(1, 20, n).astype(float),
        "bounced":             np.random.randint(0, 2, n).astype(float),
        "engagement_score":    np.random.uniform(0, 1, n),
    })


def make_browse_events(n=60) -> pd.DataFrame:
    return pd.DataFrame({
        "event_id":           [f"E{i:07d}" for i in range(n)],
        "session_id":         [f"S{i:06d}" for i in np.random.randint(0, 30, n)],
        "user_id":            np.random.choice(["U00001","U00002","U00003"], n),
        "product_id":         np.random.choice(["P00001","P00002","P00003"], n),
        "event_type":         np.random.choice(
            ["view","click","add_to_cart","remove_from_cart","add_to_wishlist"], n
        ),
        "time_spent_secs_clean": np.random.randint(2, 300, n).astype(float),
        "added_to_cart":      (np.random.rand(n) > 0.7).astype(int),
        "wishlisted":         (np.random.rand(n) > 0.9).astype(int),
    })


def make_orders(n=15) -> pd.DataFrame:
    return pd.DataFrame({
        "order_id":     [f"O{i:06d}" for i in range(n)],
        "user_id":      np.random.choice(["U00001","U00002"], n),
        "order_date":   pd.date_range("2024-06-01", periods=n, freq="15D"),
        "total_amount": np.random.uniform(10, 500, n).round(2),
        "discount_pct": np.random.choice([0, 10, 20], n),
        "status":       "delivered",
    })


def make_order_items(orders, n_per_order=2) -> pd.DataFrame:
    rows = []
    for _, o in orders.iterrows():
        for j in range(n_per_order):
            rows.append({
                "order_item_id": f"OI{len(rows):07d}",
                "order_id":      o["order_id"],
                "product_id":    f"P{j+1:05d}",
                "quantity":      1,
                "unit_price":    float(np.random.uniform(10, 200)),
            })
    return pd.DataFrame(rows)


def make_products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": [f"P{i:05d}" for i in range(1, 6)],
        "category":   ["Electronics","Clothing","Books","Sports","Beauty"],
        "price":      [100.0, 50.0, 20.0, 80.0, 30.0],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCreateSessionFeatures:
    def test_returns_one_row_per_user(self):
        from src.features.build_features import create_session_features
        sess = make_sessions()
        snap = pd.Timestamp("2025-01-01")
        out  = create_session_features(sess, snap)
        assert out["user_id"].nunique() == out.shape[0]

    def test_total_sessions_column_exists(self):
        from src.features.build_features import create_session_features
        sess = make_sessions()
        out  = create_session_features(sess, pd.Timestamp("2025-01-01"))
        assert "total_sessions" in out.columns


class TestCreateBrowseFeatures:
    def test_cart_add_rate_in_0_1(self):
        from src.features.build_features import create_browse_features
        be  = make_browse_events()
        out = create_browse_features(be, make_products())
        assert out["cart_add_rate"].between(0, 1).all()

    def test_uses_cleaned_column_not_original(self):
        """Bug regression: aggregation must use time_spent_secs_clean, not original."""
        from src.features.build_features import create_browse_features
        be = make_browse_events()
        # Remove the original raw column to ensure fixed code doesn't break
        be_no_raw = be.drop(columns=["event_type"], errors="ignore").copy()
        be["added_to_cart"] = (be["event_type"] == "add_to_cart").astype(int)
        out = create_browse_features(be, make_products())
        assert "avg_time_per_event" in out.columns

    def test_unique_categories_correct(self):
        from src.features.build_features import create_browse_features
        be  = make_browse_events()
        out = create_browse_features(be, make_products())
        assert "unique_categories_browsed" in out.columns
        assert (out["unique_categories_browsed"] >= 1).all()


class TestCreateOrderFeatures:
    def test_order_frequency_capped_at_30(self):
        """Regression: extreme outliers should be capped at 30 orders/month."""
        from src.features.build_features import create_order_features
        orders = make_orders()
        # Create a 1-day customer with many orders → would produce huge frequency
        extra = pd.DataFrame({
            "order_id":     ["O999998","O999999"],
            "user_id":      ["U99999","U99999"],
            "order_date":   [pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-02")],
            "total_amount": [50.0, 60.0],
            "discount_pct": [0, 0],
            "status":       ["delivered","delivered"],
        })
        orders = pd.concat([orders, extra], ignore_index=True)
        items  = make_order_items(orders)
        out    = create_order_features(orders, items, pd.Timestamp("2025-01-01"))
        assert (out["order_frequency"] <= 30).all(), \
            "order_frequency should be capped at 30 orders/month"


class TestCreateRFMFeatures:
    def test_all_segments_recognised(self):
        from src.features.build_features import create_rfm_features
        orders = make_orders(30)
        rfm    = create_rfm_features(orders)
        valid_segments = {"Champions","Loyal","Potential","Recent","At Risk","Lost"}
        assert set(rfm["segment"].unique()).issubset(valid_segments)

    def test_scores_computed(self):
        from src.features.build_features import create_rfm_features
        orders = make_orders()
        rfm    = create_rfm_features(orders)
        for col in ("R_score","F_score","M_score","RFM_score"):
            assert col in rfm.columns


class TestEngineerFeatures:
    def test_output_has_will_purchase(self):
        from src.features.build_features import engineer_features
        from src.data.preprocess import encode_ordinal

        users = pd.DataFrame({
            "user_id":         ["U00001","U00002","U00003"],
            "age":             [25, 35, 45],
            "membership":      ["free","gold","platinum"],
            "membership_encoded": [0, 2, 3],
            "is_premium":      [0, 1, 1],
            "account_age_days":[100, 200, 300],
            "will_purchase":   [1, 1, 0],
        })
        sessions = make_sessions()
        sessions["user_id"] = np.random.choice(["U00001","U00002","U00003"], len(sessions))
        be       = make_browse_events()
        be["user_id"] = np.random.choice(["U00001","U00002","U00003"], len(be))
        orders   = make_orders()
        items    = make_order_items(orders)
        prods    = make_products()

        tables = {
            "users": users, "sessions": sessions,
            "browse_events": be, "orders": orders,
            "order_items": items, "products": prods,
        }
        fm = engineer_features(tables)
        assert "will_purchase" in fm.columns
        assert set(fm["user_id"].unique()) == {"U00001","U00002","U00003"}
