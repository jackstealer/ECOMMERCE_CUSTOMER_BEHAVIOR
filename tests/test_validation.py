"""Tests for src/data/validation.py"""
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.validation import DataValidator


class TestDataFreshnessCheck:
    def test_fresh_data_passes(self):
        validator = DataValidator()
        # Create data with max date = reference - 3 days
        ref = datetime(2025, 1, 10)
        df  = pd.DataFrame({"date_col": [
            "2025-01-07", "2025-01-08", "2025-01-09",
        ]})
        # Should pass with max_age_days=7 and reference 10 Jan
        ok = validator.check_data_freshness(
            df, "date_col", max_age_days=7, reference_date=ref
        )
        assert ok is True

    def test_stale_data_fails(self):
        validator = DataValidator()
        ref = datetime(2026, 6, 1)   # current date
        df  = pd.DataFrame({"date_col": ["2025-05-31"]})
        ok = validator.check_data_freshness(
            df, "date_col", max_age_days=7, reference_date=ref
        )
        assert ok is False

    def test_historic_synthetic_data_with_injected_reference(self):
        """
        Regression: using datetime.now() caused synthetic 2025 data to always fail.
        With reference_date=END_DATE+1 the check should pass.
        """
        validator = DataValidator()
        end_date = datetime(2025, 5, 31)
        ref      = datetime(2025, 6, 1)   # one day after end
        df = pd.DataFrame({"date_col": [
            "2025-05-28", "2025-05-29", "2025-05-30", "2025-05-31",
        ]})
        ok = validator.check_data_freshness(
            df, "date_col", max_age_days=7, reference_date=ref
        )
        assert ok is True


class TestReferentialIntegrity:
    def test_clean_fk_passes(self):
        validator = DataValidator()
        parent = pd.DataFrame({"user_id": ["U001","U002","U003"]})
        child  = pd.DataFrame({
            "order_id": ["O1","O2"],
            "user_id":  ["U001","U002"],
        })
        ok = validator.check_referential_integrity(
            child, parent, "user_id", "user_id", "orders", "users"
        )
        assert ok is True
        assert len(validator.validation_errors) == 0

    def test_orphan_records_detected(self):
        validator = DataValidator()
        parent = pd.DataFrame({"user_id": ["U001","U002"]})
        child  = pd.DataFrame({
            "order_id": ["O1","O2"],
            "user_id":  ["U001","U999"],   # U999 doesn't exist in parent
        })
        ok = validator.check_referential_integrity(
            child, parent, "user_id", "user_id", "orders", "users"
        )
        assert ok is False
        assert len(validator.validation_errors) == 1
        assert "U999" in str(validator.validation_errors[0])

    def test_missing_fk_column_warns(self):
        validator = DataValidator()
        parent = pd.DataFrame({"id": ["A"]})
        child  = pd.DataFrame({"other_col": [1]})   # no FK column
        ok = validator.check_referential_integrity(
            child, parent, "user_id", "id"
        )
        assert ok is True   # skipped, not an error
        assert len(validator.warnings) == 1


class TestCheckMissingValues:
    def test_passes_below_threshold(self):
        validator = DataValidator()
        df = pd.DataFrame({"a": [1, 2, np.nan], "b": [1.0, 2.0, 3.0]})
        ok = validator.check_missing_values(df, max_missing_pct=0.5)
        assert ok is True

    def test_fails_above_threshold(self):
        validator = DataValidator()
        df = pd.DataFrame({"a": [np.nan, np.nan, 1.0]})   # 2/3 = 66% missing
        ok = validator.check_missing_values(df, max_missing_pct=0.5)
        assert ok is False


class TestCheckDuplicates:
    def test_detects_duplicates(self):
        validator = DataValidator()
        df = pd.DataFrame({"a": [1, 1, 2]})
        ok = validator.check_duplicates(df)
        assert ok is False

    def test_no_duplicates_passes(self):
        validator = DataValidator()
        df = pd.DataFrame({"a": [1, 2, 3]})
        ok = validator.check_duplicates(df)
        assert ok is True
