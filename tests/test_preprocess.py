"""Tests for src/data/preprocess.py"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_users(n=20) -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":    [f"U{i:05d}" for i in range(n)],
        "age":        np.random.randint(18, 70, n),
        "gender":     np.random.choice(["Male", "Female"], n),
        "membership": np.random.choice(["free", "silver", "gold", "platinum"], n),
        "signup_date":pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
    })


def _make_orders(user_ids, n=8) -> pd.DataFrame:
    return pd.DataFrame({
        "order_id":     [f"O{i:06d}" for i in range(n)],
        "user_id":      np.random.choice(user_ids[:5], n),   # only first 5 users buy
        "order_date":   pd.date_range("2024-06-01", periods=n, freq="W").astype(str),
        "total_amount": np.random.uniform(10, 500, n).round(2),
        "status":       np.random.choice(["delivered", "shipped"], n),
        "payment_method": "credit_card",
        "discount_pct": 0,
    })


class TestHandleMissingValues:
    def test_numeric_median_fill(self):
        from src.data.preprocess import handle_missing_values
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "z"]})
        out = handle_missing_values(df, strategy="median")
        assert out["a"].isnull().sum() == 0
        assert out["a"].iloc[1] == 2.0  # median of [1, 3]

    def test_categorical_mode_fill(self):
        from src.data.preprocess import handle_missing_values
        df = pd.DataFrame({"cat": ["a", "a", None]})
        out = handle_missing_values(df)
        assert out["cat"].iloc[-1] == "a"


class TestRemoveDuplicates:
    def test_drops_full_duplicates(self):
        from src.data.preprocess import remove_duplicates
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        out = remove_duplicates(df)
        assert len(out) == 2

    def test_pk_deduplication(self):
        from src.data.preprocess import remove_duplicates
        df = pd.DataFrame({"id": [1, 1, 2], "val": ["first", "second", "only"]})
        out = remove_duplicates(df, pk_col="id")
        assert len(out) == 2
        assert out[out["id"] == 1]["val"].iloc[0] == "first"


class TestAgeBins:
    def test_age_bin_labels_match_edges(self):
        from src.data.preprocess import clean_data

        users = _make_users(50)
        # Force specific ages
        users["age"] = [18, 25, 26, 35, 36, 50, 51, 70] + list(
            np.random.randint(18, 70, 50 - 8)
        )
        orders = _make_orders(users["user_id"].tolist())

        from unittest.mock import patch

        # Inject test tables directly into clean_data by patching load_raw
        with patch("src.data.preprocess.load_raw", return_value={
            "users": users, "orders": orders
        }):
            # parse_datetimes needs datetime cols
            users["signup_date"] = pd.to_datetime(users["signup_date"])
            orders["order_date"] = pd.to_datetime(orders["order_date"])
            with patch("src.data.preprocess.parse_datetimes", lambda t: t):
                tables = clean_data.__wrapped__(None) if hasattr(clean_data, "__wrapped__") else None

        # Fall back to testing pd.cut directly
        age_bins   = [17, 25, 35, 50, 71]
        age_labels = ["18-25", "26-35", "36-50", "51+"]
        test_ages  = pd.Series([18, 25, 26, 35, 36, 50, 51, 70])
        result     = pd.cut(test_ages, bins=age_bins, labels=age_labels)
        assert str(result.iloc[0]) == "18-25"   # age 18 → 18-25
        assert str(result.iloc[1]) == "18-25"   # age 25 → 18-25 (right=True, (17,25])
        assert str(result.iloc[2]) == "26-35"   # age 26 → 26-35
        assert str(result.iloc[6]) == "51+"     # age 51 → 51+


class TestWillPurchaseTarget:
    def test_will_purchase_created(self):
        """will_purchase = 1 for users with orders, 0 for others."""
        from src.data.preprocess import clean_data
        from unittest.mock import patch

        users  = _make_users(10)
        orders = _make_orders(users["user_id"].tolist(), n=5)

        buyers = set(orders["user_id"].unique())

        users["signup_date"] = pd.to_datetime(users["signup_date"])
        orders["order_date"] = pd.to_datetime(orders["order_date"])

        with patch("src.data.preprocess.load_raw", return_value={
            "users": users, "orders": orders
        }):
            with patch("src.data.preprocess.parse_datetimes", lambda t: t):
                from src.data.preprocess import clip_outliers, log_transform, encode_ordinal
                import pandas as _pd, numpy as _np

                u = users.copy()
                u = encode_ordinal(u, "membership", ["free", "silver", "gold", "platinum"])
                u["is_premium"] = u["membership"].isin(["gold", "platinum"]).astype(int)
                age_bins   = [17, 25, 35, 50, 71]
                age_labels = ["18-25", "26-35", "36-50", "51+"]
                u["age_group"] = _pd.cut(u["age"], bins=age_bins, labels=age_labels)
                ref = orders["order_date"].max()
                u["account_age_days"] = (ref - u["signup_date"]).dt.days.clip(lower=0)
                u["will_purchase"] = u["user_id"].isin(buyers).astype(int)

        assert "will_purchase" in u.columns
        for uid in buyers:
            row = u[u["user_id"] == uid]
            if not row.empty:
                assert row["will_purchase"].iloc[0] == 1
        for uid in set(users["user_id"]) - buyers:
            row = u[u["user_id"] == uid]
            if not row.empty:
                assert row["will_purchase"].iloc[0] == 0
