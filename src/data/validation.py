"""
Data validation and quality checks.

Changes vs original:
  - Fixed check_data_freshness to compare timezone-naive datetimes consistently
  - Added check_referential_integrity for foreign-key validation
  - Freshness check now accepts a reference date so tests can inject a fixed date
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Validate data quality and schema."""

    def __init__(self):
        self.validation_errors: List[str] = []
        self.warnings: List[str] = []

    def validate_schema(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that DataFrame has required columns.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Returns:
            True if valid, False otherwise
        """
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False

        logger.info("Schema validation passed")
        return True

    def check_missing_values(self, df: pd.DataFrame, max_missing_pct: float = 0.5) -> bool:
        """
        Check for excessive missing values.

        Args:
            df: DataFrame to check
            max_missing_pct: Maximum allowed missing percentage (0-1)

        Returns:
            True if acceptable, False otherwise
        """
        missing_pct = df.isnull().sum() / len(df)
        problematic_cols = missing_pct[missing_pct > max_missing_pct]

        if not problematic_cols.empty:
            error_msg = (
                f"Columns with >{max_missing_pct*100}% missing: "
                f"{problematic_cols.to_dict()}"
            )
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False

        # Log warnings for columns with some missing values
        some_missing = missing_pct[(missing_pct > 0) & (missing_pct <= max_missing_pct)]
        if not some_missing.empty:
            warning_msg = f"Columns with missing values: {some_missing.to_dict()}"
            logger.warning(warning_msg)
            self.warnings.append(warning_msg)

        return True

    def check_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> bool:
        """
        Check for duplicate rows.

        Args:
            df: DataFrame to check
            subset: Columns to check for duplicates

        Returns:
            True if no duplicates, False otherwise
        """
        n_duplicates = df.duplicated(subset=subset).sum()

        if n_duplicates > 0:
            warning_msg = f"Found {n_duplicates} duplicate rows"
            logger.warning(warning_msg)
            self.warnings.append(warning_msg)
            return False

        return True

    def check_data_types(
        self, df: pd.DataFrame, expected_types: Dict[str, str]
    ) -> bool:
        """
        Validate column data types.

        Args:
            df: DataFrame to check
            expected_types: Dict of column_name: expected_type

        Returns:
            True if types match, False otherwise
        """
        type_mismatches = []

        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    type_mismatches.append(
                        f"{col}: expected {expected_type}, got {actual_type}"
                    )

        if type_mismatches:
            error_msg = f"Data type mismatches: {type_mismatches}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False

        return True

    def check_value_ranges(
        self, df: pd.DataFrame, ranges: Dict[str, Tuple[float, float]]
    ) -> bool:
        """
        Check if numeric values are within expected ranges.

        Args:
            df: DataFrame to check
            ranges: Dict of column_name: (min_value, max_value)

        Returns:
            True if all values in range, False otherwise
        """
        out_of_range = []

        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                out_of_range_count = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range_count > 0:
                    out_of_range.append(
                        f"{col}: {out_of_range_count} values out of "
                        f"range [{min_val}, {max_val}]"
                    )

        if out_of_range:
            error_msg = f"Values out of range: {out_of_range}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False

        return True

    def check_data_freshness(
        self,
        df: pd.DataFrame,
        date_column: str,
        max_age_days: int = 7,
        reference_date: Optional[datetime] = None,
    ) -> bool:
        """
        Check if data is fresh (not too old).

        Args:
            df: DataFrame to check
            date_column: Column containing dates
            max_age_days: Maximum allowed age in days
            reference_date: Comparison reference (defaults to datetime.utcnow()).
                            Inject a fixed date in tests to avoid spurious failures.

        Returns:
            True if data is fresh, False otherwise
        """
        if date_column not in df.columns:
            return True

        try:
            dates = pd.to_datetime(df[date_column])
            # Strip any timezone info to ensure naive comparison
            dates = dates.dt.tz_localize(None) if dates.dt.tz is not None else dates
            max_date = dates.max()

            ref = reference_date or datetime.utcnow()
            # Ensure ref is also naive
            if hasattr(ref, "tzinfo") and ref.tzinfo is not None:
                ref = ref.replace(tzinfo=None)

            days_old = (ref - max_date).days

            if days_old > max_age_days:
                warning_msg = (
                    f"Data is {days_old} days old "
                    f"(max allowed: {max_age_days} days). "
                    f"Latest record: {max_date.date()}"
                )
                logger.warning(warning_msg)
                self.warnings.append(warning_msg)
                return False

            logger.info(f"Data freshness check passed: {days_old} days old")
            return True

        except Exception as e:
            logger.error(f"Failed to check data freshness: {e}")
            return False

    def check_referential_integrity(
        self,
        child_df: pd.DataFrame,
        parent_df: pd.DataFrame,
        child_fk: str,
        parent_pk: str,
        child_table: str = "child",
        parent_table: str = "parent",
    ) -> bool:
        """
        Validate that all foreign key values in child_df exist in parent_df.

        Args:
            child_df:     DataFrame containing the foreign key
            parent_df:    DataFrame containing the primary key
            child_fk:     Foreign key column name in child_df
            parent_pk:    Primary key column name in parent_df
            child_table:  Human-readable name for logging
            parent_table: Human-readable name for logging

        Returns:
            True if no orphan records found, False otherwise
        """
        if child_fk not in child_df.columns or parent_pk not in parent_df.columns:
            warning_msg = (
                f"Referential integrity check skipped: "
                f"{child_fk} or {parent_pk} column not found"
            )
            logger.warning(warning_msg)
            self.warnings.append(warning_msg)
            return True

        parent_ids = set(parent_df[parent_pk].unique())
        orphan_mask = ~child_df[child_fk].isin(parent_ids)
        n_orphans = orphan_mask.sum()

        if n_orphans > 0:
            orphan_ids = child_df.loc[orphan_mask, child_fk].unique()[:5]
            error_msg = (
                f"Referential integrity violation: {n_orphans} rows in "
                f"'{child_table}'.{child_fk} reference non-existent "
                f"'{parent_table}'.{parent_pk} (e.g. {list(orphan_ids)})"
            )
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False

        logger.info(
            f"Referential integrity OK: {child_table}.{child_fk} → "
            f"{parent_table}.{parent_pk}"
        )
        return True

    def generate_quality_report(self, df: pd.DataFrame, table_name: str) -> Dict:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze
            table_name: Name of the table

        Returns:
            Quality report dictionary
        """
        logger.info(f"Generating quality report for {table_name}")

        report = {
            "table_name":     table_name,
            "total_rows":     len(df),
            "total_columns":  len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "data_types":     df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "numeric_summary":     {},
            "categorical_summary": {},
        }

        # Numeric columns summary
        for col in df.select_dtypes(include=[np.number]).columns:
            report["numeric_summary"][col] = {
                "mean":       float(df[col].mean()),
                "std":        float(df[col].std()),
                "min":        float(df[col].min()),
                "max":        float(df[col].max()),
                "null_count": int(df[col].isnull().sum()),
            }

        # Categorical columns summary
        for col in df.select_dtypes(include=["object"]).columns:
            report["categorical_summary"][col] = {
                "unique_values": int(df[col].nunique()),
                "most_common":   df[col].value_counts().head(3).to_dict(),
                "null_count":    int(df[col].isnull().sum()),
            }

        logger.info(f"Quality report generated for {table_name}")
        return report

    def validate_all(
        self,
        df: pd.DataFrame,
        table_name: str,
        required_columns: List[str] = None,
    ) -> Tuple[bool, Dict]:
        """
        Run all validation checks.

        Args:
            df: DataFrame to validate
            table_name: Name of the table
            required_columns: List of required columns

        Returns:
            (is_valid, quality_report)
        """
        logger.info(f"Starting validation for {table_name}")
        self.validation_errors = []
        self.warnings = []

        checks = []

        if required_columns:
            checks.append(self.validate_schema(df, required_columns))

        checks.append(self.check_missing_values(df))
        self.check_duplicates(df)

        quality_report = self.generate_quality_report(df, table_name)
        quality_report["validation_errors"] = self.validation_errors
        quality_report["warnings"]          = self.warnings

        is_valid = all(checks) and len(self.validation_errors) == 0

        if is_valid:
            logger.info(f"✓ Validation passed for {table_name}")
        else:
            logger.error(f"✗ Validation failed for {table_name}")
            logger.error(f"Errors: {self.validation_errors}")

        if self.warnings:
            logger.warning(f"Warnings: {self.warnings}")

        return is_valid, quality_report
