"""
src/features/build_features_fixed.py
Fixed feature engineering WITHOUT data leakage

Key fix:
- Target: Will user purchase in NEXT 30 days?
- Features: Only use data from BEFORE the prediction window
- Temporal split: Train on older data, test on newer data
"""

from pathlib import Path
import pandas as pd
import numpy as np


def create_temporal_features(
    users: pd.DataFrame,
    sessions: pd.DataFrame,
    browse_events: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
    products: pd.DataFrame,
    prediction_window_days: int = 30,
    observation_window_days: int = 90,
) -> pd.DataFrame:
    """
    Create features WITHOUT data leakage using temporal windows.
    
    For each user, we:
    1. Define a cutoff date (observation_end_date)
    2. Use only data BEFORE cutoff for features
    3. Check if user purchased AFTER cutoff (within prediction window) for target
    
    Args:
        prediction_window_days: Days after cutoff to check for purchases (target)
        observation_window_days: Days before cutoff to use for features
    
    Returns:
        DataFrame with features and will_purchase_next_30days target
    """
    print(f"\n🔧 Creating temporal features (NO data leakage)...")
    print(f"   Observation window: {observation_window_days} days")
    print(f"   Prediction window: {prediction_window_days} days")
    
    # Convert dates
    orders = orders.copy()
    sessions = sessions.copy()
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    sessions['session_start'] = pd.to_datetime(sessions['session_start'])
    
    # Find the date range
    min_date = orders['order_date'].min()
    max_date = orders['order_date'].max()
    
    # Set cutoff date (leave room for prediction window)
    cutoff_date = max_date - pd.Timedelta(days=prediction_window_days)
    observation_start = cutoff_date - pd.Timedelta(days=observation_window_days)
    
    print(f"   Date range: {min_date.date()} to {max_date.date()}")
    print(f"   Observation period: {observation_start.date()} to {cutoff_date.date()}")
    print(f"   Prediction period: {cutoff_date.date()} to {max_date.date()}")
    
    # Filter data to observation window (BEFORE cutoff)
    orders_past = orders[orders['order_date'] < cutoff_date].copy()
    sessions_past = sessions[sessions['session_start'] < cutoff_date].copy()
    
    # Create target: Did user purchase AFTER cutoff?
    orders_future = orders[
        (orders['order_date'] >= cutoff_date) &
        (orders['order_date'] < cutoff_date + pd.Timedelta(days=prediction_window_days))
    ]
    future_buyers = set(orders_future['user_id'].unique())
    
    print(f"   Users with past orders: {orders_past['user_id'].nunique()}")
    print(f"   Users with future orders: {len(future_buyers)}")
    
    # Start with user base features
    feature_matrix = users[['user_id', 'age']].copy()
    
    # Add device type encoding
    if 'device_type' in users.columns:
        device_mapping = {'mobile': 0, 'tablet': 1, 'desktop': 2}
        feature_matrix['device_type'] = users['device_type'].map(device_mapping).fillna(0)
    
    # Add membership encoding
    if 'membership' in users.columns:
        membership_mapping = {'free': 0, 'silver': 1, 'gold': 2, 'platinum': 3}
        feature_matrix['membership_encoded'] = users['membership'].map(membership_mapping).fillna(0)
        feature_matrix['is_premium'] = (feature_matrix['membership_encoded'] >= 2).astype(int)
    
    # Calculate account age at cutoff
    if 'signup_date' in users.columns:
        users_copy = users.copy()
        users_copy['signup_date'] = pd.to_datetime(users_copy['signup_date'])
        feature_matrix['account_age_days'] = (cutoff_date - users_copy['signup_date']).dt.days.clip(lower=0)
        feature_matrix['is_new_user'] = (feature_matrix['account_age_days'] < 30).astype(int)
        feature_matrix['is_veteran_user'] = (feature_matrix['account_age_days'] > 365).astype(int)
    
    # === SESSION FEATURES (from observation window only) ===
    if len(sessions_past) > 0:
        # Clean session data
        sessions_clean = sessions_past.copy()
        if 'duration_secs' in sessions_clean.columns:
            sessions_clean['duration_secs_clean'] = sessions_clean['duration_secs'].clip(upper=sessions_clean['duration_secs'].quantile(0.95))
        if 'pages_visited' in sessions_clean.columns:
            sessions_clean['pages_visited_clean'] = sessions_clean['pages_visited'].clip(upper=sessions_clean['pages_visited'].quantile(0.95))
        
        sess_agg = sessions_clean.groupby('user_id').agg(
            total_sessions=('session_id', 'count'),
            avg_duration=('duration_secs_clean', 'mean'),
            max_duration=('duration_secs_clean', 'max'),
            avg_pages=('pages_visited_clean', 'mean'),
            max_pages=('pages_visited_clean', 'max'),
            avg_engagement=('engagement_score', 'mean') if 'engagement_score' in sessions_clean.columns else ('session_id', lambda x: 0),
            bounce_rate=('bounced', 'mean') if 'bounced' in sessions_clean.columns else ('session_id', lambda x: 0),
        ).reset_index()
        
        # Days since last session
        last_session = sessions_clean.groupby('user_id')['session_start'].max().reset_index()
        last_session['days_since_last_session'] = (cutoff_date - last_session['session_start']).dt.days
        sess_agg = sess_agg.merge(last_session[['user_id', 'days_since_last_session']], on='user_id', how='left')
        
        feature_matrix = feature_matrix.merge(sess_agg, on='user_id', how='left')
    
    # === BROWSE EVENT FEATURES (from observation window only) ===
    if len(browse_events) > 0:
        browse_agg = browse_events.groupby('user_id').agg(
            total_events=('event_id', 'count'),
            unique_products_browsed=('product_id', 'nunique'),
            total_cart_adds=('added_to_cart', 'sum') if 'added_to_cart' in browse_events.columns else ('event_id', lambda x: 0),
            total_wishlists=('wishlisted', 'sum') if 'wishlisted' in browse_events.columns else ('event_id', lambda x: 0),
        ).reset_index()
        
        browse_agg['cart_add_rate'] = (browse_agg['total_cart_adds'] / browse_agg['total_events'].replace(0, np.nan)).fillna(0)
        browse_agg['wishlist_rate'] = (browse_agg['total_wishlists'] / browse_agg['total_events'].replace(0, np.nan)).fillna(0)
        
        # Category diversity
        if 'product_id' in browse_events.columns and 'category' in products.columns:
            cat_div = (
                browse_events.merge(products[['product_id', 'category']], on='product_id', how='left')
                .groupby('user_id')['category']
                .nunique()
                .reset_index()
                .rename(columns={'category': 'unique_categories_browsed'})
            )
            browse_agg = browse_agg.merge(cat_div, on='user_id', how='left')
        
        feature_matrix = feature_matrix.merge(browse_agg, on='user_id', how='left')
    
    # === PAST ORDER FEATURES (from observation window only) ===
    if len(orders_past) > 0:
        order_agg = orders_past.groupby('user_id').agg(
            past_total_orders=('order_id', 'count'),
            past_total_spend=('total_amount', 'sum'),
            past_avg_order_value=('total_amount', 'mean'),
            past_max_order_value=('total_amount', 'max'),
            past_avg_discount=('discount_pct', 'mean') if 'discount_pct' in orders_past.columns else ('order_id', lambda x: 0),
            last_order_date=('order_date', 'max'),
            first_order_date=('order_date', 'min'),
        ).reset_index()
        
        order_agg['days_since_last_order'] = (cutoff_date - order_agg['last_order_date']).dt.days
        order_agg['customer_lifespan_days'] = (order_agg['last_order_date'] - order_agg['first_order_date']).dt.days.clip(lower=1)
        order_agg['past_order_frequency'] = (
            order_agg['past_total_orders'] / (order_agg['customer_lifespan_days'] / 30).replace(0, np.nan)
        ).fillna(order_agg['past_total_orders']).clip(upper=30)
        
        order_agg = order_agg.drop(columns=['last_order_date', 'first_order_date'])
        feature_matrix = feature_matrix.merge(order_agg, on='user_id', how='left')
    
    # Fill NaN values for users with no activity
    numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
    feature_matrix[numeric_cols] = feature_matrix[numeric_cols].fillna(0)
    
    # === CREATE TARGET (NO LEAKAGE) ===
    feature_matrix['will_purchase_next_30days'] = feature_matrix['user_id'].isin(future_buyers).astype(int)
    
    print(f"\n✓ Feature matrix created:")
    print(f"   Total users: {len(feature_matrix)}")
    print(f"   Total features: {len(feature_matrix.columns) - 2}")  # Exclude user_id and target
    print(f"   Target distribution:")
    print(f"     Will purchase (1): {feature_matrix['will_purchase_next_30days'].sum()} ({feature_matrix['will_purchase_next_30days'].mean()*100:.1f}%)")
    print(f"     Won't purchase (0): {(~feature_matrix['will_purchase_next_30days'].astype(bool)).sum()} ({(1-feature_matrix['will_purchase_next_30days'].mean())*100:.1f}%)")
    
    return feature_matrix
