"""
Feature engineering pipeline for customer behavior data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_rfm_features(df):
    """
    Create RFM (Recency, Frequency, Monetary) features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with customer data
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with RFM features added
    """
    df_features = df.copy()
    
    # Recency score (inverse of days_since_last_purchase)
    df_features['recency_score'] = pd.cut(
        df_features['days_since_last_purchase'],
        bins=[0, 30, 90, 180, float('inf')],
        labels=[5, 3, 2, 1]
    ).astype(int)
    
    # Frequency score
    df_features['frequency_score'] = pd.cut(
        df_features['total_purchases'],
        bins=[0, 2, 5, 10, float('inf')],
        labels=[1, 2, 3, 5]
    ).astype(int)
    
    # Monetary score
    df_features['monetary_score'] = pd.cut(
        df_features['total_spent'],
        bins=[0, 100, 300, 500, float('inf')],
        labels=[1, 2, 3, 5]
    ).astype(int)
    
    # Overall RFM score
    df_features['rfm_score'] = (
        df_features['recency_score'] + 
        df_features['frequency_score'] + 
        df_features['monetary_score']
    )
    
    return df_features


def create_engagement_features(df):
    """
    Create customer engagement features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with engagement features
    """
    df_features = df.copy()
    
    # Purchase frequency (purchases per visit)
    df_features['purchase_per_visit'] = (
        df_features['total_purchases'] / 
        np.maximum(df_features['website_visits'], 1)
    )
    
    # Engagement score
    df_features['engagement_score'] = (
        df_features['email_open_rate'] * 100 + 
        df_features['website_visits'] / 10
    )
    
    # Customer lifetime value estimate
    df_features['estimated_clv'] = (
        df_features['avg_order_value'] * 
        df_features['total_purchases'] * 
        (1 - df_features['days_since_last_purchase'] / 365)
    )
    
    return df_features


def encode_categorical_features(df):
    """
    Encode categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded categorical features
    """
    df_encoded = df.copy()
    
    # Label encode gender
    if 'gender' in df_encoded.columns:
        le = LabelEncoder()
        df_encoded['gender_encoded'] = le.fit_transform(df_encoded['gender'])
    
    return df_encoded


def engineer_features(df):
    """
    Main feature engineering pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all engineered features
    """
    print("Starting feature engineering...")
    
    # Create RFM features
    df_features = create_rfm_features(df)
    print("Created RFM features")
    
    # Create engagement features
    df_features = create_engagement_features(df_features)
    print("Created engagement features")
    
    # Encode categorical features
    df_features = encode_categorical_features(df_features)
    print("Encoded categorical features")
    
    print("Feature engineering complete!")
    return df_features
