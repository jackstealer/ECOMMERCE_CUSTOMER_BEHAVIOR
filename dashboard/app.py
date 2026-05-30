"""
Streamlit dashboard for E-Commerce Customer Behavior Analysis.
Phase 7 - Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Page configuration
st.set_page_config(
    page_title="E-Commerce Customer Behavior Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Title
st.title("🛒 E-Commerce Customer Behavior Dashboard")
st.markdown("---")

# Sidebar
st.sidebar.header("Dashboard Controls")
page = st.sidebar.selectbox(
    "Select Page",
    ["Overview", "Customer Segmentation", "Churn Analysis", "Predictions"]
)

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/processed/customer_behavior_features.csv')
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please run the notebooks first.")
        return None

df = load_data()

if df is not None:
    # Overview Page
    if page == "Overview":
        st.header("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            st.metric("Avg Purchase Value", f"${df['avg_order_value'].mean():.2f}")
        with col3:
            st.metric("Churn Rate", f"{df['churned'].mean()*100:.1f}%")
        with col4:
            st.metric("Avg Purchases", f"{df['total_purchases'].mean():.1f}")
        
        st.markdown("---")
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig = px.histogram(df, x='age', nbins=30, 
                             title="Customer Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Total Spent Distribution")
            fig = px.histogram(df, x='total_spent', nbins=30,
                             title="Customer Spending Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    # Customer Segmentation Page
    elif page == "Customer Segmentation":
        st.header("👥 Customer Segmentation")
        
        if 'rfm_score' in df.columns:
            # RFM Segmentation
            st.subheader("RFM Score Distribution")
            fig = px.histogram(df, x='rfm_score', 
                             title="RFM Score Distribution",
                             labels={'rfm_score': 'RFM Score'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot
            st.subheader("Customer Segments")
            fig = px.scatter(df, x='total_purchases', y='total_spent',
                           color='rfm_score', size='avg_order_value',
                           hover_data=['customer_id'],
                           title="Customer Segmentation by Purchase Behavior")
            st.plotly_chart(fig, use_container_width=True)
    
    # Churn Analysis Page
    elif page == "Churn Analysis":
        st.header("⚠️ Churn Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn by Gender")
            churn_gender = df.groupby('gender')['churned'].mean().reset_index()
            fig = px.bar(churn_gender, x='gender', y='churned',
                        title="Churn Rate by Gender")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Churn by Age Group")
            df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100],
                                    labels=['18-30', '31-45', '46-60', '60+'])
            churn_age = df.groupby('age_group')['churned'].mean().reset_index()
            fig = px.bar(churn_age, x='age_group', y='churned',
                        title="Churn Rate by Age Group")
            st.plotly_chart(fig, use_container_width=True)
    
    # Predictions Page
    elif page == "Predictions":
        st.header("🔮 Customer Churn Predictions")
        
        st.info("This section would display model predictions. Train a model first using notebook 05.")
        
        # Sample prediction interface
        st.subheader("Predict Churn for New Customer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 75, 35)
            total_purchases = st.number_input("Total Purchases", 0, 100, 5)
            total_spent = st.number_input("Total Spent ($)", 0.0, 5000.0, 250.0)
        
        with col2:
            days_since_last = st.number_input("Days Since Last Purchase", 0, 365, 30)
            email_open_rate = st.slider("Email Open Rate", 0.0, 1.0, 0.3)
            website_visits = st.number_input("Website Visits", 0, 100, 10)
        
        if st.button("Predict Churn"):
            st.warning("Model not loaded. Please train a model first.")

# Footer
st.markdown("---")
st.markdown("**E-Commerce Customer Behavior ML Project** | Built with Streamlit")
