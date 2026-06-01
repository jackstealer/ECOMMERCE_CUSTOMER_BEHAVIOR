"""
Phase 1 — Data Collection & Understanding

This script covers:
- Loading the raw dataset
- Initial data exploration
- Understanding data structure and types
- Identifying potential issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')

def main():
    """Main function to run data understanding analysis."""
    
    print("=" * 80)
    print("PHASE 1: DATA COLLECTION & UNDERSTANDING")
    print("=" * 80)
    
    # Load raw data
    print("\n1. Loading raw data...")
    df = pd.read_csv('../data/raw/customer_behavior.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Display first few rows
    print("\n2. First 5 rows of the dataset:")
    print(df.head())
    
    # Data info
    print("\n3. Dataset Information:")
    print(df.info())
    
    # Statistical summary
    print("\n4. Statistical Summary:")
    print(df.describe())
    
    # Check for missing values
    print("\n5. Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found!")
    
    # Check for duplicates
    print("\n6. Duplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    # Data types
    print("\n7. Data Types:")
    print(df.dtypes)
    
    # Unique values in categorical columns
    print("\n8. Unique Values in Categorical Columns:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")
        print(f"  Values: {df[col].unique()[:5]}")
    
    # Target variable distribution
    if 'churned' in df.columns:
        print("\n9. Target Variable Distribution (Churned):")
        print(df['churned'].value_counts())
        print(f"\nChurn Rate: {df['churned'].mean() * 100:.2f}%")
    
    # Basic visualizations
    print("\n10. Generating basic visualizations...")
    
    # Age distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.hist(df['age'], bins=30, edgecolor='black', alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # Total spent distribution
    plt.subplot(2, 2, 2)
    plt.hist(df['total_spent'], bins=30, edgecolor='black', alpha=0.7, color='green')
    plt.title('Total Spent Distribution')
    plt.xlabel('Total Spent ($)')
    plt.ylabel('Frequency')
    
    # Total purchases distribution
    plt.subplot(2, 2, 3)
    plt.hist(df['total_purchases'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    plt.title('Total Purchases Distribution')
    plt.xlabel('Total Purchases')
    plt.ylabel('Frequency')
    
    # Churn distribution
    if 'churned' in df.columns:
        plt.subplot(2, 2, 4)
        df['churned'].value_counts().plot(kind='bar', color=['green', 'red'], alpha=0.7)
        plt.title('Churn Distribution')
        plt.xlabel('Churned')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('../reports/figures/01_data_understanding.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to: ../reports/figures/01_data_understanding.png")
    plt.show()
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE: Data Understanding Finished!")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    df = main()
