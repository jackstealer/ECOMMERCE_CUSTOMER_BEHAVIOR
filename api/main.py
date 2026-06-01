import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import joblib
import pandas as pd
import numpy as np
import uvicorn
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score

# Make project root importable
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from config.settings import settings

PROCESSED_DIR = settings.PROCESSED_DATA_DIR
OUTPUT_DIR    = settings.OUTPUT_DIR
MODELS_DIR    = settings.MODEL_DIR

best_model = None
model_metadata = {}
feature_matrix_df = None

def load_model():
    if not MODELS_DIR.exists():
        return None, {}
    pkl_files  = sorted(MODELS_DIR.glob("best_model_*.pkl"))
    meta_files = sorted(MODELS_DIR.glob("model_metadata_*.json"))
    if not pkl_files:
        return None, {}
    model = joblib.load(pkl_files[-1])
    meta  = json.loads(meta_files[-1].read_text()) if meta_files else {}
    return model, meta

def load_feature_matrix():
    """Load the feature matrix with real customer data and encode categorical variables"""
    feature_path = PROCESSED_DIR / "feature_matrix.csv"
    if feature_path.exists():
        df = pd.read_csv(feature_path)
    else:
        # Try parquet format
        feature_path = PROCESSED_DIR / "feature_matrix.parquet"
        if feature_path.exists():
            df = pd.read_parquet(feature_path)
        else:
            return None
    
    # Encode categorical variables (handle both 'object' and 'str' dtypes)
    if 'device_type' in df.columns and df['device_type'].dtype in ['object', 'str', pd.StringDtype()]:
        device_mapping = {'mobile': 0, 'tablet': 1, 'desktop': 2}
        df['device_type'] = df['device_type'].map(device_mapping).fillna(0).astype(float)
    
    return df

def calculate_live_metrics():
    """Calculate metrics in real-time from actual data"""
    global best_model, feature_matrix_df
    
    if best_model is None or feature_matrix_df is None:
        return None
    
    # Prepare features and target (data is already encoded in load_feature_matrix)
    X = feature_matrix_df.drop(['user_id', 'will_purchase'], axis=1, errors='ignore')
    y = feature_matrix_df['will_purchase'] if 'will_purchase' in feature_matrix_df.columns else None
    
    if y is None:
        return None
    
    # Make predictions
    y_pred = best_model.predict(X)
    y_proba = best_model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = X.columns.tolist()
        
        # Get top 10 features
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i] for i in indices]
        top_importances = [float(importances[i]) for i in indices]
    else:
        top_features = []
        top_importances = []
    
    # Customer statistics
    total_customers = len(feature_matrix_df)
    high_value_customers = len(feature_matrix_df[feature_matrix_df['total_spend'] > feature_matrix_df['total_spend'].median()])
    avg_order_value = float(feature_matrix_df['avg_order_value'].mean())
    total_revenue = float(feature_matrix_df['total_spend'].sum())
    
    return {
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc_roc": float(roc_auc)
        },
        "roc_curve": [[float(x), float(y)] for x, y in zip(fpr, tpr)],
        "feature_importance": {
            "labels": top_features,
            "values": top_importances
        },
        "customer_stats": {
            "total_customers": total_customers,
            "high_value_customers": high_value_customers,
            "avg_order_value": avg_order_value,
            "total_revenue": total_revenue,
            "purchase_rate": float(y.mean()) if y is not None else 0
        },
        "data_timestamp": pd.Timestamp.now().isoformat()
    }

@asynccontextmanager
async def lifespan(app: FastAPI):
    global best_model, model_metadata, feature_matrix_df
    best_model, model_metadata = load_model()
    feature_matrix_df = load_feature_matrix()
    yield

app = FastAPI(title="E-Commerce ML API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body for prediction - all 38 features in correct order
class PredictRequest(BaseModel):
    age: float
    account_age_days: float
    device_type: float
    membership_encoded: float
    is_premium: float
    is_new_user: float
    is_veteran_user: float
    duration_secs_clean_mean: float
    duration_secs_clean_max: float
    duration_secs_clean_sum: float
    pages_visited_clean_mean: float
    pages_visited_clean_max: float
    pages_visited_clean_sum: float
    engagement_score_mean: float
    engagement_score_max: float
    engagement_score_sum: float
    bounced_mean: float
    bounced_max: float
    bounced_sum: float
    total_sessions: float
    days_since_last_session: float
    total_events: float
    unique_products_browsed: float
    avg_time_per_event: float
    total_cart_adds: float
    total_wishlists: float
    cart_add_rate: float
    wishlist_rate: float
    view_to_cart_ratio: float
    unique_categories_browsed: float
    total_orders: float
    total_spend: float
    avg_order_value: float
    max_order_value: float
    avg_discount_pct: float
    days_since_last_order: float
    customer_lifespan_days: float
    order_frequency: float

@app.get("/api/data")
def get_dashboard_data():
    """Return real-time dashboard stats calculated from live data."""
    global feature_matrix_df
    
    if feature_matrix_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature matrix not found. Run pipeline.py first."
        )
    
    # Calculate metrics in real-time
    live_data = calculate_live_metrics()
    
    if live_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to calculate metrics. Model or data not available."
        )
    
    return live_data

@app.get("/api/customers")
def get_customers(limit: int = 100):
    """Get real customer data"""
    global feature_matrix_df
    
    if feature_matrix_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer data not found."
        )
    
    # Return sample of customers
    customers = feature_matrix_df.head(limit).to_dict('records')
    return {
        "total": len(feature_matrix_df),
        "customers": customers
    }

@app.post("/api/customers")
def add_customer(req: PredictRequest):
    """Add a new customer to the dataset and recalculate metrics"""
    global feature_matrix_df
    
    if feature_matrix_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer data not found."
        )
    
    # Convert request to dict
    req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    
    # Generate new user_id
    new_user_id = f"U{len(feature_matrix_df) + 1:05d}"
    
    # Make prediction for this customer
    X_sample = pd.DataFrame([req_dict])
    will_purchase = int(best_model.predict(X_sample)[0])
    
    # Add to dataframe
    new_row = {"user_id": new_user_id, "will_purchase": will_purchase, **req_dict}
    feature_matrix_df = pd.concat([feature_matrix_df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated data
    feature_path = PROCESSED_DIR / "feature_matrix.csv"
    feature_matrix_df.to_csv(feature_path, index=False)
    
    return {
        "message": "Customer added successfully",
        "user_id": new_user_id,
        "will_purchase": will_purchase,
        "total_customers": len(feature_matrix_df)
    }

@app.post("/api/predict")
def predict(req: PredictRequest):
    global best_model, model_metadata
    if best_model is None:
        best_model, model_metadata = load_model()
        if best_model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Run pipeline.py first."
            )

    # Convert request dict to DataFrame (supports Pydantic V1 and V2)
    req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    
    # Ensure correct feature order as expected by the model
    feature_names = model_metadata.get('feature_names', [])
    if feature_names:
        # Reorder features to match training order
        ordered_data = {feat: req_dict.get(feat, 0) for feat in feature_names}
        X_sample = pd.DataFrame([ordered_data])
    else:
        X_sample = pd.DataFrame([req_dict])

    try:
        proba = float(best_model.predict_proba(X_sample)[0, 1])
        
        # Business rule: Apply recency decay for churning customers
        days_since_last_order = req_dict.get('days_since_last_order', 0)
        total_orders = req_dict.get('total_orders', 0)
        
        # Strong decay for inactive customers with order history
        if days_since_last_order > 90 and total_orders > 0:
            # Aggressive exponential decay
            # 90 days: 0.7x, 120 days: 0.5x, 150 days: 0.35x, 180 days: 0.25x, 365 days: 0.1x
            if days_since_last_order >= 180:
                decay_factor = 0.25  # Very strong penalty for 180+ days
            elif days_since_last_order >= 150:
                decay_factor = 0.35
            elif days_since_last_order >= 120:
                decay_factor = 0.5
            else:  # 90-119 days
                decay_factor = 0.7
            
            proba = proba * decay_factor
        
        # If no orders at all and inactive, reduce probability significantly
        if total_orders == 0 and days_since_last_order > 60:
            proba = proba * 0.3
        
        # Final prediction based on adjusted probability
        prediction = 1 if proba >= 0.5 else 0
        
        return {
            "purchase_probability": proba,
            "prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
