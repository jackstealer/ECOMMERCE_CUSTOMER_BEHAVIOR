"""
Phase 4 — Feature Engineering

This script covers:
- Creating new features
- Feature transformations
- Encoding categorical variables
- Feature scaling
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('../src')
from features.build_features import engineer_features, create_rfm_features

def main():
    """Main function to run feature engineering pipeline."""
    
    print("=" * 80)
    print("PHASE 4: FEATURE ENGINEERING")
    print("=" * 80)
    
    # Load cleaned data
    print("\n1. Loading cleaned data...")
    df = pd.read_csv('../data/processed/customer_behavior_clean.csv')
    print(f"Original dataset shape: {df.shape}")
    print(f"Original features: {list(df.columns)}")
    
    # Engineer features
    print("\n2. Engineering features...")
    df_features = engineer_features(df)
    
    print(f"\nFeature-engineered dataset shape: {df_features.shape}")
    print(f"New features added: {df_features.shape[1] - df.shape[1]}")
    
    # Display new features
    new_features = [col for col in df_features.columns if col not in df.columns]
    print(f"\n3. New Features Created:")
    for i, feature in enumerate(new_features, 1):
        print(f"   {i}. {feature}")
    
    # Display sample of engineered features
    print("\n4. Sample of Engineered Features:")
    feature_cols = ['recency_score', 'frequency_score', 'monetary_score', 
                   'rfm_score', 'purchase_per_visit', 'engagement_score']
    available_cols = [col for col in feature_cols if col in df_features.columns]
    
    if available_cols:
        print(df_features[available_cols].head(10))
    
    # Feature statistics
    print("\n5. Feature Statistics:")
    if 'rfm_score' in df_features.columns:
        print(f"\nRFM Score Distribution:")
        print(df_features['rfm_score'].describe())
        print(f"\nRFM Score Value Counts:")
        print(df_features['rfm_score'].value_counts().sort_index())
    
    if 'engagement_score' in df_features.columns:
        print(f"\nEngagement Score Statistics:")
        print(df_features['engagement_score'].describe())
    
    # Customer segmentation based on RFM
    if 'rfm_score' in df_features.columns:
        print("\n6. Customer Segmentation (RFM-based):")
        
        # Define segments
        df_features['customer_segment'] = pd.cut(
            df_features['rfm_score'],
            bins=[0, 5, 9, 12, 15],
            labels=['At Risk', 'Potential', 'Loyal', 'Champion']
        )
        
        segment_counts = df_features['customer_segment'].value_counts()
        print("\nCustomer Segments:")
        for segment, count in segment_counts.items():
            percentage = (count / len(df_features)) * 100
            print(f"  {segment}: {count} ({percentage:.1f}%)")
    
    # Feature correlation with target
    if 'churned' in df_features.columns:
        print("\n7. Feature Correlation with Churn:")
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        correlations = df_features[numeric_cols].corrwith(df_features['churned']).sort_values(ascending=False)
        
        print("\nTop 10 Positive Correlations:")
        print(correlations.head(10))
        
        print("\nTop 10 Negative Correlations:")
        print(correlations.tail(10))
    
    # Save feature-engineered data
    output_path = '../data/processed/customer_behavior_features.csv'
    df_features.to_csv(output_path, index=False)
    print(f"\n8. Feature-engineered data saved to: {output_path}")
    
    # Summary
    print("\n9. Feature Engineering Summary:")
    print(f"   - Original features: {df.shape[1]}")
    print(f"   - New features: {len(new_features)}")
    print(f"   - Total features: {df_features.shape[1]}")
    print(f"   - Dataset size: {df_features.shape[0]} rows")
    
    print("\n" + "=" * 80)
    print("PHASE 4 COMPLETE: Feature Engineering Finished!")
    print("=" * 80)
    
    return df_features


if __name__ == "__main__":
    df_features = main()
