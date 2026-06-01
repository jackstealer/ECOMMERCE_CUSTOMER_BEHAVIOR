"""
Phase 2 — Cleaning & Preprocessing

This script covers:
- Handling missing values
- Removing duplicates
- Data type conversions
- Outlier detection and treatment
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('../src')
from data.preprocess import clean_data, handle_missing_values, remove_duplicates, handle_outliers

def main():
    """Main function to run data cleaning pipeline."""
    
    print("=" * 80)
    print("PHASE 2: DATA CLEANING & PREPROCESSING")
    print("=" * 80)
    
    # Load raw data
    print("\n1. Loading raw data...")
    df = pd.read_csv('../data/raw/customer_behavior.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Check initial data quality
    print("\n2. Initial Data Quality Check:")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    # Display missing values by column
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nMissing values by column:")
        print(missing_values[missing_values > 0])
    
    # Clean data using preprocessing functions
    print("\n3. Applying data cleaning pipeline...")
    df_clean = clean_data(df)
    
    print(f"\nCleaned dataset shape: {df_clean.shape}")
    print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")
    
    # Verify data quality after cleaning
    print("\n4. Data Quality After Cleaning:")
    print(f"Missing values: {df_clean.isnull().sum().sum()}")
    print(f"Duplicate rows: {df_clean.duplicated().sum()}")
    
    # Check data types
    print("\n5. Data Types After Cleaning:")
    print(df_clean.dtypes)
    
    # Statistical summary after cleaning
    print("\n6. Statistical Summary After Cleaning:")
    print(df_clean.describe())
    
    # Save cleaned data
    output_path = '../data/processed/customer_behavior_clean.csv'
    df_clean.to_csv(output_path, index=False)
    print(f"\n7. Cleaned data saved to: {output_path}")
    
    # Summary statistics comparison
    print("\n8. Summary Statistics Comparison:")
    print("\nBefore Cleaning:")
    print(df.describe().loc[['mean', 'std', 'min', 'max']])
    print("\nAfter Cleaning:")
    print(df_clean.describe().loc[['mean', 'std', 'min', 'max']])
    
    print("\n" + "=" * 80)
    print("PHASE 2 COMPLETE: Data Cleaning Finished!")
    print("=" * 80)
    
    return df_clean


if __name__ == "__main__":
    df_clean = main()
