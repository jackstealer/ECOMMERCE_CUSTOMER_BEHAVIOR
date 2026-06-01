"""
dashboard/app.py
E-Commerce Customer Behavior ML — Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

from pathlib import Path
import warnings, joblib, json
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "E-Commerce ML Dashboard",
    page_icon  = "🛒",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Dynamic paths ─────────────────────────────────────────────────────────────
_script_dir   = Path(__file__).resolve().parent
_candidates   = [
    _script_dir.parent / 'data' / 'processed',
    _script_dir        / 'data' / 'processed',
    Path().resolve()   / 'data' / 'processed',
    Path().resolve().parent / 'data' / 'processed',
]
PROCESSED_DIR = next((p for p in _candidates if p.exists()), None)

if PROCESSED_DIR is None:
    st.error("❌ `data/processed/` not found. Run notebooks Phase 1–5 first.")
    st.stop()

PROJECT_ROOT = PROCESSED_DIR.parent.parent
FIGURES_DIR  = PROJECT_ROOT / 'reports' / 'figures'
MODELS_DIR   = PROJECT_ROOT / 'data' / 'outputs' / 'models'
OUTPUT_DIR   = PROJECT_ROOT / 'data' / 'outputs'

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data
def load(name):
    p = PROCESSED_DIR / f'{name}.parquet'
    if p.exists(): return pd.read_parquet(p)
    p2 = PROCESSED_DIR / f'{name}.csv'
    if p2.exists(): return pd.read_csv(p2)
    return None

@st.cache_resource
def load_model():
    files = sorted(MODELS_DIR.glob('best_model_*.pkl')) if MODELS_DIR.exists() else []
    if not files: return None, None
    model = joblib.load(files[-1])
    meta_files = sorted(MODELS_DIR.glob('model_metadata_*.json'))
    meta = json.loads(meta_files[-1].read_text()) if meta_files else {}
    return model, meta

def load_eval():
    p = OUTPUT_DIR / 'evaluation_report.csv'
    return pd.read_csv(p) if p.exists() else None

def fig_to_st(fig):
    st.pyplot(fig)
    plt.close(fig)

# ── Load data ─────────────────────────────────────────────────────────────────
users         = load('users')
sessions      = load('sessions')
orders        = load('orders')
browse_events = load('browse_events')
products      = load('products')
fm            = load('feature_matrix')
best_model, metadata = load_model()
eval_report   = load_eval()

FEATURE_COLS = [c for c in fm.columns if c not in ('user_id','will_purchase')] if fm is not None else []
TARGET_COL   = 'will_purchase'

# ── Sidebar navigation ────────────────────────────────────────────────────────
PAGES = {
    "🏠 Overview":         "overview",
    "📊 Data Insights":    "eda",
    "🤖 Model Performance":"model",
    "🔮 Live Predictor":   "predictor",
    "💼 Business Impact":  "business",
}

st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=60)
st.sidebar.title("E-Commerce ML")
st.sidebar.markdown("---")
page_label = st.sidebar.radio("Navigate", list(PAGES.keys()))
page       = PAGES[page_label]

st.sidebar.markdown("---")
st.sidebar.markdown("**Project phases**")
for ph in ["✅ Phase 1 — Data Understanding",
           "✅ Phase 2 — Data Cleaning",
           "✅ Phase 3 — EDA",
           "✅ Phase 4 — Feature Engineering",
           "✅ Phase 5 — Modeling",
           "✅ Phase 6 — Evaluation",
           "✅ Phase 7 — Dashboard"]:
    st.sidebar.markdown(ph)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "overview":
    st.title("🛒 E-Commerce Customer Behavior ML")
    st.markdown(
        "Predict which users are likely to purchase using machine learning "
        "on browsing history, session data, and purchase records."
    )
    st.markdown("---")

    # KPI cards
    n_users    = len(users)   if users   is not None else "N/A"
    n_orders   = len(orders)  if orders  is not None else "N/A"
    n_products = len(products)if products is not None else "N/A"
    n_events   = len(browse_events) if browse_events is not None else "N/A"
    buyer_rate = (orders['user_id'].nunique() / len(users) * 100) if (users is not None and orders is not None) else 0
    auc_val    = metadata.get('test_auc', 'N/A') if metadata else 'N/A'
    f1_val     = metadata.get('test_f1',  'N/A') if metadata else 'N/A'
    model_type = metadata.get('model_type','N/A') if metadata else 'N/A'

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Users",     f"{n_users:,}"    if isinstance(n_users,int)    else n_users)
    c2.metric("📦 Total Orders",    f"{n_orders:,}"   if isinstance(n_orders,int)   else n_orders)
    c3.metric("🛍️ Products",        f"{n_products:,}" if isinstance(n_products,int) else n_products)
    c4.metric("🖱️ Browse Events",   f"{n_events:,}"   if isinstance(n_events,int)   else n_events)

    st.markdown("")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("🎯 Buyer Rate",      f"{buyer_rate:.1f}%")
    c6.metric("📈 Model AUC-ROC",   str(auc_val))
    c7.metric("⚖️ Model F1",        str(f1_val))
    c8.metric("🤖 Model Type",      model_type)

    st.markdown("---")

    # Dataset summary
    st.subheader("📋 Dataset Summary")
    tables_info = []
    for name in ['users','products','sessions','browse_events','orders','order_items']:
        df = load(name)
        if df is not None:
            tables_info.append({'Table': name, 'Rows': f"{len(df):,}",
                                 'Columns': df.shape[1],
                                 'Null cells': df.isnull().sum().sum()})
    if tables_info:
        st.dataframe(pd.DataFrame(tables_info), use_container_width=True)

    # Evaluation report
    if eval_report is not None:
        st.markdown("---")
        st.subheader("🏆 Model Evaluation Report")
        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.dataframe(eval_report, use_container_width=True)
        with col_r:
            fig, ax = plt.subplots(figsize=(6, 3))
            numeric = eval_report[eval_report['value'] <= 1.0]
            ax.barh(numeric['metric'], numeric['value'],
                    color=sns.color_palette('Set2', len(numeric)))
            ax.set_xlim(0, 1.1)
            ax.set_xlabel('Score')
            ax.set_title('Evaluation Metrics (0–1 scale)')
            plt.tight_layout()
            fig_to_st(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "eda":
    st.title("📊 Data Insights")
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Users", "🖱️ Sessions", "📦 Products", "🏷️ RFM Segments"])

    with tab1:
        st.subheader("User Demographics")
        if users is not None:
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                users['membership'].value_counts().reindex(
                    [m for m in ['free','silver','gold','platinum'] if m in users['membership'].values]
                ).plot(kind='bar', ax=ax, edgecolor='white',
                       color=sns.color_palette('Set2',4))
                ax.set_title('Users by membership tier')
                ax.tick_params(axis='x', rotation=0)
                plt.tight_layout()
                fig_to_st(fig)
            with c2:
                fig, ax = plt.subplots(figsize=(6, 4))
                users['age'].plot(kind='hist', bins=25, ax=ax,
                                  edgecolor='white', color='#66c2a5')
                ax.axvline(users['age'].mean(), color='red', linestyle='--',
                           label=f"Mean: {users['age'].mean():.0f}")
                ax.set_title('Age distribution')
                ax.legend()
                plt.tight_layout()
                fig_to_st(fig)

            if 'age_group' in users.columns:
                fig, ax = plt.subplots(figsize=(10, 3))
                users['age_group'].value_counts().sort_index().plot(
                    kind='bar', ax=ax, edgecolor='white',
                    color=sns.color_palette('Set2',4))
                ax.set_title('Users by age group')
                ax.tick_params(axis='x', rotation=0)
                plt.tight_layout()
                fig_to_st(fig)

    with tab2:
        st.subheader("Session Behaviour")
        if sessions is not None:
            sessions['session_start'] = pd.to_datetime(sessions['session_start'])
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sessions.groupby(sessions['session_start'].dt.to_period('M')).size().plot(
                    kind='line', marker='o', ax=ax, color='#66c2a5')
                ax.set_title('Sessions per month')
                ax.tick_params(axis='x', rotation=30)
                plt.tight_layout()
                fig_to_st(fig)
            with c2:
                if 'hour_of_day' in sessions.columns:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sessions.groupby('hour_of_day').size().plot(
                        kind='bar', ax=ax, edgecolor='white', color='#fc8d62')
                    ax.set_title('Sessions by hour of day')
                    ax.tick_params(axis='x', rotation=0)
                    plt.tight_layout()
                    fig_to_st(fig)

    with tab3:
        st.subheader("Product Performance")
        if products is not None and browse_events is not None:
            cat_browse = (browse_events
                .merge(products[['product_id','category']], on='product_id', how='left')
                .groupby('category').size().sort_values(ascending=True))
            fig, ax = plt.subplots(figsize=(10, 4))
            cat_browse.plot(kind='barh', ax=ax, color='#8da0cb', edgecolor='white')
            ax.set_title('Browse events by category')
            plt.tight_layout()
            fig_to_st(fig)

            st.markdown("**Top 10 Most Browsed Products**")
            top_b = (browse_events.groupby('product_id').size()
                     .reset_index(name='browse_count')
                     .merge(products[['product_id','product_name','category','price']], on='product_id')
                     .sort_values('browse_count', ascending=False).head(10))
            st.dataframe(top_b[['product_name','category','price','browse_count']].reset_index(drop=True),
                         use_container_width=True)

    with tab4:
        st.subheader("RFM Customer Segments")
        if orders is not None:
            orders['order_date'] = pd.to_datetime(orders['order_date'])
            snap = orders['order_date'].max() + pd.Timedelta(days=1)
            rfm  = orders.groupby('user_id').agg(
                recency   =('order_date', lambda x:(snap-x.max()).days),
                frequency =('order_id','count'),
                monetary  =('total_amount','sum'),
            ).reset_index()
            rfm['R'] = pd.qcut(rfm['recency'], q=4, labels=[4,3,2,1]).astype(int)
            rfm['F'] = pd.qcut(rfm['frequency'].rank(method='first'),q=4,labels=[1,2,3,4]).astype(int)
            rfm['M'] = pd.qcut(rfm['monetary'], q=4, labels=[1,2,3,4]).astype(int)
            rfm['score'] = rfm['R'] + rfm['F'] + rfm['M']
            def seg(r):
                if r['score']>=10:              return 'Champions'
                elif r['R']>=3 and r['F']>=3:  return 'Loyal'
                elif r['R']>=3:                 return 'Recent'
                elif r['F']>=3:                 return 'At Risk'
                elif r['score']<=5:             return 'Lost'
                else:                           return 'Potential'
            rfm['segment'] = rfm.apply(seg, axis=1)

            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                rfm['segment'].value_counts().plot(kind='bar', ax=ax, edgecolor='white',
                    color=sns.color_palette('Set2', rfm['segment'].nunique()))
                ax.set_title('Customers per segment')
                ax.tick_params(axis='x', rotation=20)
                plt.tight_layout()
                fig_to_st(fig)
            with c2:
                seg_stats = rfm.groupby('segment')['monetary'].mean().sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(6, 4))
                seg_stats.plot(kind='barh', ax=ax, color='#a6d854', edgecolor='white')
                ax.set_title('Avg spend per segment ($)')
                plt.tight_layout()
                fig_to_st(fig)

            st.dataframe(
                rfm.groupby('segment').agg(
                    count=('user_id','count'),
                    avg_spend=('monetary','mean'),
                    avg_orders=('frequency','mean'),
                    avg_recency=('recency','mean')
                ).round(1).sort_values('avg_spend', ascending=False),
                use_container_width=True
            )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "model":
    st.title("🤖 Model Performance")

    if best_model is None:
        st.warning("No trained model found. Run Phase 5 notebook first.")
    else:
        if metadata:
            st.subheader("Model Details")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Model",    metadata.get('model_type','N/A'))
            c2.metric("AUC-ROC",  str(metadata.get('test_auc','N/A')))
            c3.metric("F1 Score", str(metadata.get('test_f1','N/A')))
            c4.metric("Features", str(metadata.get('n_features','N/A')))
            st.markdown("---")

        # Load saved charts from Phase 5 & 6
        for fname, title in [
            ('05_model_evaluation.png', 'Confusion Matrix | ROC Curve | Model Comparison'),
            ('05_feature_importance.png','Feature Importances'),
            ('06_roc_pr_threshold.png', 'ROC | Precision-Recall | Threshold Analysis'),
            ('06_lift_curve.png',       'Lift Curve & Cumulative Gains'),
            ('06_shap_summary.png',     'SHAP Global Explainability'),
            ('06_shap_bar.png',         'SHAP Feature Importance'),
            ('06_shap_local.png',       'SHAP Local Explanations'),
        ]:
            img_path = FIGURES_DIR / fname
            if img_path.exists():
                st.subheader(title)
                st.image(str(img_path), use_container_width=True)
                st.markdown("")

        if metadata and 'best_params' in metadata:
            st.subheader("Best Hyperparameters")
            params_df = pd.DataFrame([
                {'Parameter': k, 'Value': v}
                for k, v in metadata['best_params'].items()
            ])
            st.dataframe(params_df, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "predictor":
    st.title("🔮 Live Purchase Predictor")
    st.markdown("Adjust the sliders below to simulate a user profile and get a real-time purchase probability.")

    if best_model is None or fm is None:
        st.warning("Model or feature matrix not found — run Phase 5 first.")
    else:
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.subheader("User Profile")
            age              = st.slider("Age",                    18, 70, 30)
            membership_enc   = st.selectbox("Membership tier",
                                            [0,1,2,3],
                                            format_func=lambda x: ['Free','Silver','Gold','Platinum'][x])
            is_premium       = 1 if membership_enc >= 2 else 0
            account_age_days = st.slider("Account age (days)",    0, 1000, 180)
            is_new_user      = 1 if account_age_days < 30 else 0
            is_veteran_user  = 1 if account_age_days > 365 else 0
            age_group_enc    = 0 if age<=25 else (1 if age<=35 else (2 if age<=50 else 3))

            st.subheader("Session Behaviour")
            total_sessions          = st.slider("Total sessions",        1, 100, 10)
            duration_mean           = st.slider("Avg session duration (s)", 30, 3600, 300)
            pages_mean              = st.slider("Avg pages per session",  1, 30, 5)
            bounce_mean             = st.slider("Bounce rate",            0.0, 1.0, 0.3, step=0.05)
            days_since_last_session = st.slider("Days since last session",0, 180, 14)
            engagement_mean         = st.slider("Avg engagement score",  0.0, 1.0, 0.4, step=0.05)

            st.subheader("Browse Behaviour")
            total_events             = st.slider("Total browse events",    1, 500, 40)
            unique_products_browsed  = st.slider("Unique products viewed", 1, 200, 20)
            unique_categories        = st.slider("Unique categories",      1, 6,   3)
            cart_add_rate            = st.slider("Cart-add rate",         0.0, 0.5, 0.10, step=0.01)
            view_to_cart_ratio       = st.slider("View-to-cart ratio",    0.0, 0.5, 0.08, step=0.01)

        with col_r:
            st.subheader("Prediction")

            # Build input row matching FEATURE_COLS
            X_sample = pd.DataFrame(0.0, index=[0], columns=FEATURE_COLS)
            mapping = {
                'age': age, 'account_age_days': account_age_days,
                'membership_encoded': membership_enc,
                'is_premium': is_premium, 'age_group_enc': age_group_enc,
                'is_new_user': is_new_user, 'is_veteran_user': is_veteran_user,
                'total_sessions': total_sessions,
                'duration_secs_clean_mean': duration_mean,
                'pages_visited_clean_mean': pages_mean,
                'bounced_mean': bounce_mean,
                'days_since_last_session': days_since_last_session,
                'engagement_score_mean': engagement_mean,
                'total_events': total_events,
                'unique_products_browsed': unique_products_browsed,
                'unique_categories_browsed': unique_categories,
                'cart_add_rate': cart_add_rate,
                'view_to_cart_ratio': view_to_cart_ratio,
            }
            for col, val in mapping.items():
                if col in X_sample.columns:
                    X_sample[col] = float(val)

            proba = best_model.predict_proba(X_sample)[0, 1]
            pred  = int(proba >= 0.5)

            # Gauge-style display
            color = "#2ecc71" if proba >= 0.6 else ("#f39c12" if proba >= 0.4 else "#e74c3c")
            st.markdown(f"""
            <div style='text-align:center; padding:30px; border-radius:16px;
                        background:{color}22; border: 2px solid {color}'>
                <h1 style='color:{color}; font-size:3rem; margin:0'>{proba*100:.1f}%</h1>
                <h3 style='color:{color}; margin:8px 0'>Purchase Probability</h3>
                <p style='font-size:1.1rem; color:#555'>
                    {'🟢 Likely to purchase' if pred==1 else '🔴 Unlikely to purchase'}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            risk = "High" if proba >= 0.6 else ("Medium" if proba >= 0.4 else "Low")
            st.metric("Risk level",          risk)
            st.metric("Recommended action",
                      "Send promo email" if proba >= 0.5 else "Nurture campaign")

            # Top contributing features (simple ranking)
            st.markdown("---")
            st.subheader("Key drivers for this prediction")
            feature_contribs = {
                'Cart-add rate':           cart_add_rate * 3,
                'Total sessions':          min(total_sessions / 100, 1),
                'Engagement score':        engagement_mean,
                'Membership tier':         membership_enc / 3,
                'Account age':             min(account_age_days / 500, 1),
                'Unique categories':       unique_categories / 6,
                'Days since last session': max(0, 1 - days_since_last_session/180),
            }
            contrib_df = pd.Series(feature_contribs).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_bar = ['#2ecc71' if v >= 0 else '#e74c3c' for v in contrib_df]
            contrib_df.plot(kind='barh', ax=ax, color=colors_bar)
            ax.set_title('Feature contribution (indicative)')
            ax.set_xlabel('Signal strength')
            ax.axvline(0, color='black', lw=0.8)
            plt.tight_layout()
            fig_to_st(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — BUSINESS IMPACT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "business":
    st.title("💼 Business Impact Analysis")

    if fm is None or best_model is None:
        st.warning("Feature matrix or model not found — run Phase 4 & 5 first.")
    else:
        from sklearn.model_selection import train_test_split as tts
        X = fm[FEATURE_COLS]
        y = fm[TARGET_COL]
        X_train, X_test, y_train, y_test = tts(X, y, test_size=0.20,
                                                 stratify=y, random_state=42)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        lift_df = (pd.DataFrame({'y_true': y_test.values, 'y_score': y_proba})
                   .sort_values('y_score', ascending=False).reset_index(drop=True))
        n_total  = len(lift_df)
        n_buyers = lift_df['y_true'].sum()
        lift_df['cum_buyers']   = lift_df['y_true'].cumsum()
        lift_df['pct_targeted'] = (lift_df.index + 1) / n_total
        lift_df['pct_captured'] = lift_df['cum_buyers'] / n_buyers
        lift_df['lift']         = lift_df['pct_captured'] / lift_df['pct_targeted']

        st.subheader("⚙️ Campaign Assumptions")
        c1, c2, c3 = st.columns(3)
        avg_order  = c1.number_input("Avg order value ($)", 10.0, 500.0, 50.0, step=5.0)
        cost_pp    = c2.number_input("Cost per targeted user ($)", 0.1, 5.0, 0.5, step=0.1)
        uplift     = c3.number_input("Conversion uplift (fraction)", 0.05, 0.50, 0.15, step=0.01)

        st.markdown("---")
        st.subheader("📊 Targeting Strategy Comparison")
        rows = []
        for pct in [0.10, 0.20, 0.30, 0.40, 0.50, 1.00]:
            row      = lift_df[lift_df['pct_targeted'] >= pct].iloc[0]
            n_tgt    = int(n_total * pct)
            n_cap    = int(row['cum_buyers'])
            revenue  = n_cap * avg_order * uplift
            cost     = n_tgt * cost_pp
            profit   = revenue - cost
            roi      = profit / cost * 100 if cost > 0 else 0
            rows.append({
                'Strategy':         f"Top {pct*100:.0f}%",
                'Users targeted':   f"{n_tgt:,}",
                'Buyers captured':  f"{n_cap:,}",
                '% Buyers found':   f"{row['pct_captured']*100:.1f}%",
                'Lift':             f"{row['lift']:.2f}x",
                'Revenue ($)':      f"${revenue:,.0f}",
                'Cost ($)':         f"${cost:,.0f}",
                'Profit ($)':       f"${profit:,.0f}",
                'ROI (%)':          f"{roi:.0f}%",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("---")
        st.subheader("📈 ROI by Targeting Depth")
        roi_data = []
        for pct in np.arange(0.05, 1.01, 0.05):
            row     = lift_df[lift_df['pct_targeted'] >= pct - 0.001].iloc[0]
            n_cap   = int(row['cum_buyers'])
            revenue = n_cap * avg_order * uplift
            cost    = int(n_total * pct) * cost_pp
            roi     = (revenue - cost) / cost * 100 if cost > 0 else 0
            roi_data.append({'pct': pct*100, 'roi': roi,
                              'revenue': revenue, 'cost': cost})
        roi_df = pd.DataFrame(roi_data)

        c1, c2 = st.columns(2)
        with c1:
            best_pct = roi_df.loc[roi_df['roi'].idxmax()]
            fig, ax  = plt.subplots(figsize=(6, 4))
            ax.plot(roi_df['pct'], roi_df['roi'], color='#66c2a5', lw=2)
            ax.axhline(0, color='red', linestyle='--', lw=1)
            ax.scatter(best_pct['pct'], best_pct['roi'], color='red', s=100, zorder=5,
                       label=f"Best: top {best_pct['pct']:.0f}% → {best_pct['roi']:.0f}% ROI")
            ax.set_title('Campaign ROI by Targeting Depth')
            ax.set_xlabel('% Users Targeted')
            ax.set_ylabel('ROI (%)')
            ax.legend()
            plt.tight_layout()
            fig_to_st(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(roi_df['pct'], roi_df['revenue'], color='#66c2a5', lw=2, label='Revenue')
            ax.plot(roi_df['pct'], roi_df['cost'],    color='#fc8d62', lw=2, label='Cost')
            ax.fill_between(roi_df['pct'], roi_df['revenue'], roi_df['cost'],
                            where=roi_df['revenue'] > roi_df['cost'],
                            alpha=0.15, color='#66c2a5', label='Profit zone')
            ax.set_title('Revenue vs Cost')
            ax.set_xlabel('% Users Targeted')
            ax.set_ylabel('USD ($)')
            ax.legend()
            plt.tight_layout()
            fig_to_st(fig)

        st.markdown("---")
        st.subheader("🏆 Top Predicted Buyers")
        st.markdown("Users most likely to purchase — ranked by model score.")
        top_users = (pd.DataFrame({'user_id': fm['user_id'],
                                   'purchase_probability': best_model.predict_proba(fm[FEATURE_COLS])[:,1]})
                     .sort_values('purchase_probability', ascending=False)
                     .head(20).reset_index(drop=True))
        top_users.index += 1
        top_users['purchase_probability'] = (top_users['purchase_probability']*100).round(2).astype(str) + '%'
        top_users['recommended_action'] = top_users.index.map(
            lambda i: '🎯 Priority email' if i<=5 else ('📧 Standard email' if i<=15 else '📱 Push notification'))
        st.dataframe(top_users, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem'>"
    "E-Commerce Customer Behavior ML Project • Built with Python, scikit-learn & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
