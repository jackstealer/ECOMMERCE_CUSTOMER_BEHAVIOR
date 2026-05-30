"""
Reusable data cleaning and preprocessing functions.
"""

import pandas as pd
import numpy as np


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Strategy for handling missing values ('mean', 'median', 'drop')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with missing values handled
    """
    df_clean = df.copy()
    
    if strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    return df_clean


def remove_duplicates(df, subset=None):
    """
    Remove duplicate rows from the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    subset : list, optional
        Columns to consider for identifying duplicates
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with duplicates removed
    """
    return df.drop_duplicates(subset=subset, keep='first')


def handle_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Detect and handle outliers in specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        Columns to check for outliers
    method : str
        Method for outlier detection ('iqr', 'zscore')
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers handled
    """
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'zscore':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            df_clean[col] = df_clean[col].clip(
                lower=mean - threshold * std,
                upper=mean + threshold * std
            )
    
    return df_clean


def clean_data(df):
    """
    Main cleaning pipeline that applies all preprocessing steps.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("Starting data cleaning pipeline...")
    
    # Remove duplicates
    df_clean = remove_duplicates(df, subset=['customer_id'])
    print(f"Removed {len(df) - len(df_clean)} duplicate rows")
    
    # Handle missing values
    df_clean = handle_missing_values(df_clean, strategy='mean')
    print("Handled missing values")
    
    # Handle outliers in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'customer_id' in numeric_cols:
        numeric_cols.remove('customer_id')
    df_clean = handle_outliers(df_clean, numeric_cols, method='iqr')
    print("Handled outliers")
    
    print("Data cleaning complete!")
    return df_clean
