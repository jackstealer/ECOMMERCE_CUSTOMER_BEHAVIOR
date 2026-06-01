"""
Phase 3 — Exploratory Data Analysis

This script covers:
- Distribution analysis
- Correlation analysis
- Customer segmentation insights
- Behavioral patterns visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def main():
    """Main function to run exploratory data analysis."""
    
    print("=" * 80)
    print("PHASE 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Load cleaned data
    print("\n1. Loading cleaned data...")
    df = pd.read_csv('../data/processed/customer_behavior_clean.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Correlation analysis
    print("\n2. Correlation Analysis:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../reports/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved correlation heatmap to: ../reports/figures/correlation_heatmap.png")
    plt.show()
    
    # Distribution analysis
    print("\n3. Distribution Analysis:")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Distribution of Key Features', fontsize=16, fontweight='bold')
    
    # Age distribution
    axes[0, 0].hist(df['age'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Frequency')
    
    # Total purchases
    axes[0, 1].hist(df['total_purchases'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Total Purchases Distribution')
    axes[0, 1].set_xlabel('Total Purchases')
    axes[0, 1].set_ylabel('Frequency')
    
    # Total spent
    axes[0, 2].hist(df['total_spent'], bins=30, edgecolor='black', alpha=0.7, color='salmon')
    axes[0, 2].set_title('Total Spent Distribution')
    axes[0, 2].set_xlabel('Total Spent ($)')
    axes[0, 2].set_ylabel('Frequency')
    
    # Average order value
    axes[1, 0].hist(df['avg_order_value'], bins=30, edgecolor='black', alpha=0.7, color='gold')
    axes[1, 0].set_title('Average Order Value Distribution')
    axes[1, 0].set_xlabel('Avg Order Value ($)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Days since last purchase
    axes[1, 1].hist(df['days_since_last_purchase'], bins=30, edgecolor='black', alpha=0.7, color='plum')
    axes[1, 1].set_title('Days Since Last Purchase')
    axes[1, 1].set_xlabel('Days')
    axes[1, 1].set_ylabel('Frequency')
    
    # Email open rate
    axes[1, 2].hist(df['email_open_rate'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    axes[1, 2].set_title('Email Open Rate Distribution')
    axes[1, 2].set_xlabel('Email Open Rate')
    axes[1, 2].set_ylabel('Frequency')
    
    # Website visits
    axes[2, 0].hist(df['website_visits'], bins=30, edgecolor='black', alpha=0.7, color='lightyellow')
    axes[2, 0].set_title('Website Visits Distribution')
    axes[2, 0].set_xlabel('Website Visits')
    axes[2, 0].set_ylabel('Frequency')
    
    # Gender distribution
    if 'gender' in df.columns:
        gender_counts = df['gender'].value_counts()
        axes[2, 1].bar(gender_counts.index, gender_counts.values, alpha=0.7, color=['blue', 'pink', 'purple'])
        axes[2, 1].set_title('Gender Distribution')
        axes[2, 1].set_xlabel('Gender')
        axes[2, 1].set_ylabel('Count')
        axes[2, 1].tick_params(axis='x', rotation=45)
    
    # Churn distribution
    if 'churned' in df.columns:
        churn_counts = df['churned'].value_counts()
        axes[2, 2].bar(['Not Churned', 'Churned'], churn_counts.values, 
                      alpha=0.7, color=['green', 'red'])
        axes[2, 2].set_title('Churn Distribution')
        axes[2, 2].set_xlabel('Status')
        axes[2, 2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('../reports/figures/feature_distributions.png', dpi=300, bbox_inches='tight')
    print("Saved feature distributions to: ../reports/figures/feature_distributions.png")
    plt.show()
    
    # Churn analysis by demographics
    if 'churned' in df.columns:
        print("\n4. Churn Analysis by Demographics:")
        
        # Churn by gender
        if 'gender' in df.columns:
            churn_by_gender = df.groupby('gender')['churned'].mean() * 100
            print("\nChurn Rate by Gender:")
            print(churn_by_gender)
        
        # Churn by age group
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                                labels=['18-30', '31-45', '46-60', '60+'])
        churn_by_age = df.groupby('age_group')['churned'].mean() * 100
        print("\nChurn Rate by Age Group:")
        print(churn_by_age)
        
        # Visualize churn analysis
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        if 'gender' in df.columns:
            churn_by_gender.plot(kind='bar', ax=axes[0], color=['blue', 'pink', 'purple'], alpha=0.7)
            axes[0].set_title('Churn Rate by Gender', fontweight='bold')
            axes[0].set_xlabel('Gender')
            axes[0].set_ylabel('Churn Rate (%)')
            axes[0].tick_params(axis='x', rotation=45)
        
        churn_by_age.plot(kind='bar', ax=axes[1], color='orange', alpha=0.7)
        axes[1].set_title('Churn Rate by Age Group', fontweight='bold')
        axes[1].set_xlabel('Age Group')
        axes[1].set_ylabel('Churn Rate (%)')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../reports/figures/churn_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSaved churn analysis to: ../reports/figures/churn_analysis.png")
        plt.show()
    
    # Key insights
    print("\n5. Key Insights:")
    print(f"- Average customer age: {df['age'].mean():.1f} years")
    print(f"- Average total spent: ${df['total_spent'].mean():.2f}")
    print(f"- Average purchases per customer: {df['total_purchases'].mean():.1f}")
    print(f"- Average order value: ${df['avg_order_value'].mean():.2f}")
    if 'churned' in df.columns:
        print(f"- Overall churn rate: {df['churned'].mean() * 100:.2f}%")
    
    print("\n" + "=" * 80)
    print("PHASE 3 COMPLETE: Exploratory Data Analysis Finished!")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    df = main()
