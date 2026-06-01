import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { Activity, Target, TrendingUp, Zap, Award, BarChart2, RefreshCw, Users, DollarSign, ShoppingCart } from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const API_BASE = 'http://127.0.0.1:8000/api';

// Animated Counter Component
const AnimatedCounter = ({ value, suffix = '', duration = 2000 }) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let startTime;
    const startValue = 0;
    const endValue = parseFloat(value) || 0;

    const animate = (currentTime) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      setCount(startValue + (endValue - startValue) * easeOutQuart);

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    requestAnimationFrame(animate);
  }, [value, duration]);

  return <span>{count.toFixed(1)}{suffix}</span>;
};

function App() {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  
  const [formData, setFormData] = useState({
    age: 0,
    account_age_days: 0,
    device_type: 0,
    membership_encoded: 0,
    is_premium: 0,
    is_new_user: 0,
    is_veteran_user: 0,
    duration_secs_clean_mean: 0,
    duration_secs_clean_max: 0,
    duration_secs_clean_sum: 0,
    pages_visited_clean_mean: 0,
    pages_visited_clean_max: 0,
    pages_visited_clean_sum: 0,
    engagement_score_mean: 0,
    engagement_score_max: 0,
    engagement_score_sum: 0,
    bounced_mean: 0,
    bounced_max: 0,
    bounced_sum: 0,
    total_sessions: 0,
    days_since_last_session: 0,
    total_events: 0,
    unique_products_browsed: 0,
    avg_time_per_event: 0,
    total_cart_adds: 0,
    total_wishlists: 0,
    cart_add_rate: 0,
    wishlist_rate: 0,
    view_to_cart_ratio: 0,
    unique_categories_browsed: 0,
    total_orders: 0,
    total_spend: 0,
    avg_order_value: 0,
    max_order_value: 0,
    avg_discount_pct: 0,
    days_since_last_order: 0,
    customer_lifespan_days: 0,
    order_frequency: 0,
  });

  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    setRefreshing(true);
    try {
      const res = await axios.get(API_BASE + '/data');
      setDashboardData(res.data);
      setLastUpdated(new Date());
      setLoading(false);
    } catch (err) {
      console.warn('Dashboard data not found.');
      setLoading(false);
    }
    setRefreshing(false);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setPredicting(true);
    try {
      const res = await axios.post(API_BASE + '/predict', formData);
      setPrediction(res.data);
    } catch (err) {
      console.error(err);
      alert('Prediction failed. Check API connection.');
    }
    setPredicting(false);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: parseFloat(value) || 0
    }));
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: { 
        position: 'bottom', 
        labels: { 
          color: '#94a3b8',
          padding: 15,
          font: { size: 11, family: 'Inter' },
          usePointStyle: true,
          pointStyle: 'circle'
        }
      },
      tooltip: {
        backgroundColor: 'rgba(15, 23, 42, 0.95)',
        padding: 12,
        titleColor: '#f1f5f9',
        bodyColor: '#cbd5e1',
        borderColor: '#3b82f6',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += context.parsed.y.toFixed(3);
            }
            return label;
          }
        }
      }
    },
    scales: {
      x: { 
        grid: { 
          color: 'rgba(148, 163, 184, 0.1)',
          drawBorder: false
        },
        ticks: { 
          color: '#94a3b8', 
          font: { size: 10 }
        }
      },
      y: { 
        grid: { 
          color: 'rgba(148, 163, 184, 0.1)',
          drawBorder: false
        },
        ticks: { 
          color: '#94a3b8', 
          font: { size: 10 }
        }
      }
    },
    interaction: {
      mode: 'index',
      intersect: false,
    },
    animation: {
      duration: 2000,
      easing: 'easeInOutQuart'
    }
  };

  return (
    <div className="dashboard-container">
      <header>
        <div className="logo">
          <h1>🚀 Customer Analytics Platform</h1>
          <span>Enterprise ML-Powered Insights • Real-time Predictions</span>
        </div>
        <button 
          onClick={fetchDashboardData} 
          disabled={refreshing}
          style={{
            padding: '0.75rem 1.5rem',
            background: 'linear-gradient(135deg, #3b82f6 0%, #2563eb 100%)',
            border: 'none',
            borderRadius: '8px',
            color: 'white',
            cursor: refreshing ? 'not-allowed' : 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            fontSize: '0.9rem',
            fontWeight: '600',
            transition: 'all 0.3s ease',
            opacity: refreshing ? 0.6 : 1
          }}
        >
          <RefreshCw size={18} style={{animation: refreshing ? 'spin 1s linear infinite' : 'none'}} />
          {refreshing ? 'Refreshing...' : 'Refresh Data'}
        </button>
      </header>

      {lastUpdated && (
        <div style={{
          textAlign: 'center',
          color: '#94a3b8',
          fontSize: '0.85rem',
          marginBottom: '1rem',
          padding: '0.5rem',
          background: 'rgba(59, 130, 246, 0.1)',
          borderRadius: '6px'
        }}>
          📊 Live Data • Last Updated: {lastUpdated.toLocaleTimeString()}
          {dashboardData?.customer_stats && (
            <span style={{marginLeft: '1rem'}}>
              • {dashboardData.customer_stats.total_customers.toLocaleString()} Total Customers
            </span>
          )}
        </div>
      )}

      <section className="metrics-grid">
        <div className="metric-card">
          <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
            <Award size={20} color="#3b82f6" />
            <div className="metric-title">Model Accuracy</div>
          </div>
          <div className="metric-value">
            {dashboardData?.metrics?.accuracy ? (
              <AnimatedCounter value={dashboardData.metrics.accuracy * 100} suffix="%" />
            ) : '--'}
          </div>
          <div style={{fontSize: '0.75rem', color: '#10b981', marginTop: '0.5rem'}}>
            <TrendingUp size={14} style={{display: 'inline', marginRight: '4px'}} />
            Real-time calculation
          </div>
        </div>
        
        <div className="metric-card">
          <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
            <Target size={20} color="#10b981" />
            <div className="metric-title">Precision</div>
          </div>
          <div className="metric-value">
            {dashboardData?.metrics?.precision ? (
              <AnimatedCounter value={dashboardData.metrics.precision * 100} suffix="%" />
            ) : '--'}
          </div>
          <div style={{fontSize: '0.75rem', color: '#10b981', marginTop: '0.5rem'}}>
            <TrendingUp size={14} style={{display: 'inline', marginRight: '4px'}} />
            Live metrics
          </div>
        </div>
        
        <div className="metric-card">
          <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
            <Zap size={20} color="#f59e0b" />
            <div className="metric-title">Recall</div>
          </div>
          <div className="metric-value">
            {dashboardData?.metrics?.recall ? (
              <AnimatedCounter value={dashboardData.metrics.recall * 100} suffix="%" />
            ) : '--'}
          </div>
          <div style={{fontSize: '0.75rem', color: '#10b981', marginTop: '0.5rem'}}>
            <TrendingUp size={14} style={{display: 'inline', marginRight: '4px'}} />
            Dynamic data
          </div>
        </div>
        
        <div className="metric-card">
          <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
            <BarChart2 size={20} color="#8b5cf6" />
            <div className="metric-title">AUC-ROC Score</div>
          </div>
          <div className="metric-value">
            {dashboardData?.metrics?.auc_roc ? (
              <AnimatedCounter value={dashboardData.metrics.auc_roc} suffix="" />
            ) : '--'}
          </div>
          <div style={{fontSize: '0.75rem', color: '#10b981', marginTop: '0.5rem'}}>
            <TrendingUp size={14} style={{display: 'inline', marginRight: '4px'}} />
            Real-time score
          </div>
        </div>
      </section>

      {dashboardData?.customer_stats && (
        <section className="metrics-grid" style={{marginTop: '1rem'}}>
          <div className="metric-card">
            <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
              <Users size={20} color="#06b6d4" />
              <div className="metric-title">Total Customers</div>
            </div>
            <div className="metric-value" style={{fontSize: '2rem'}}>
              {dashboardData.customer_stats.total_customers.toLocaleString()}
            </div>
            <div style={{fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem'}}>
              Active customer base
            </div>
          </div>

          <div className="metric-card">
            <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
              <Award size={20} color="#f59e0b" />
              <div className="metric-title">High-Value Customers</div>
            </div>
            <div className="metric-value" style={{fontSize: '2rem'}}>
              {dashboardData.customer_stats.high_value_customers.toLocaleString()}
            </div>
            <div style={{fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem'}}>
              Above median spend
            </div>
          </div>

          <div className="metric-card">
            <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
              <ShoppingCart size={20} color="#10b981" />
              <div className="metric-title">Avg Order Value</div>
            </div>
            <div className="metric-value" style={{fontSize: '2rem'}}>
              ${dashboardData.customer_stats.avg_order_value.toFixed(2)}
            </div>
            <div style={{fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem'}}>
              Per transaction
            </div>
          </div>

          <div className="metric-card">
            <div style={{display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem'}}>
              <DollarSign size={20} color="#8b5cf6" />
              <div className="metric-title">Total Revenue</div>
            </div>
            <div className="metric-value" style={{fontSize: '2rem'}}>
              ${(dashboardData.customer_stats.total_revenue / 1000).toFixed(1)}K
            </div>
            <div style={{fontSize: '0.75rem', color: '#94a3b8', marginTop: '0.5rem'}}>
              Cumulative sales
            </div>
          </div>
        </section>
      )}

      {dashboardData && (
        <section className="charts-grid">
          <div className="chart-panel">
            <h2>📈 ROC Curve Analysis</h2>
            <Line
              options={chartOptions}
              data={{
                datasets: [
                  {
                    label: 'Model Performance',
                    data: dashboardData.roc_curve.map(p => ({ x: p[0], y: p[1] })),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    fill: true,
                    tension: 0.4,
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    pointHoverBackgroundColor: '#3b82f6',
                    pointHoverBorderColor: '#fff',
                    pointHoverBorderWidth: 2,
                  },
                  {
                    label: 'Random Baseline',
                    data: [{x: 0, y: 0}, {x: 1, y: 1}],
                    borderColor: '#64748b',
                    borderDash: [5, 5],
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                  }
                ]
              }}
            />
            <div style={{marginTop: '1rem', padding: '1rem', background: 'rgba(59, 130, 246, 0.1)', borderRadius: '8px', fontSize: '0.875rem'}}>
              <strong style={{color: '#3b82f6'}}>AUC Score: {dashboardData.metrics.auc_roc?.toFixed(3)}</strong>
              <span style={{color: '#94a3b8', marginLeft: '1rem'}}>• Excellent model discrimination</span>
            </div>
          </div>
          
          <div className="chart-panel">
            <h2>🎯 Top 10 Feature Importance</h2>
            <Bar
              options={{ 
                ...chartOptions, 
                indexAxis: 'y',
                plugins: {
                  ...chartOptions.plugins,
                  legend: { display: false }
                }
              }}
              data={{
                labels: dashboardData.feature_importance.labels.slice(0, 10),
                datasets: [
                  {
                    label: 'Importance Score',
                    data: dashboardData.feature_importance.values.slice(0, 10),
                    backgroundColor: [
                      'rgba(59, 130, 246, 0.8)',
                      'rgba(16, 185, 129, 0.8)',
                      'rgba(245, 158, 11, 0.8)',
                      'rgba(139, 92, 246, 0.8)',
                      'rgba(236, 72, 153, 0.8)',
                      'rgba(6, 182, 212, 0.8)',
                      'rgba(132, 204, 22, 0.8)',
                      'rgba(249, 115, 22, 0.8)',
                      'rgba(99, 102, 241, 0.8)',
                      'rgba(20, 184, 166, 0.8)'
                    ],
                    borderRadius: 6,
                    borderWidth: 0,
                    barThickness: 20,
                  }
                ]
              }}
            />
            <div style={{marginTop: '1rem', padding: '1rem', background: 'rgba(16, 185, 129, 0.1)', borderRadius: '8px', fontSize: '0.875rem'}}>
              <strong style={{color: '#10b981'}}>Key Drivers Identified</strong>
              <span style={{color: '#94a3b8', marginLeft: '1rem'}}>• Order frequency is the strongest predictor</span>
            </div>
          </div>
        </section>
      )}

      <section className="predictor-section">
        <div>
          <h2>🎯 Real-Time Purchase Prediction</h2>
          <form onSubmit={handlePredict}>
            <div style={{display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem'}}>
              <div className="form-group">
                <label>👤 Customer Age</label>
                <input type="number" name="age" value={formData.age} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>📅 Account Age (days)</label>
                <input type="number" name="account_age_days" value={formData.account_age_days} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>🔄 Total Sessions</label>
                <input type="number" name="total_sessions" value={formData.total_sessions} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>⏱️ Avg Session Duration (s)</label>
                <input type="number" name="duration_secs_clean_mean" value={formData.duration_secs_clean_mean} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>📦 Total Orders</label>
                <input type="number" name="total_orders" value={formData.total_orders} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>💰 Total Spend ($)</label>
                <input type="number" name="total_spend" value={formData.total_spend} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>💵 Avg Order Value ($)</label>
                <input type="number" name="avg_order_value" value={formData.avg_order_value} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>🛒 Cart Add Rate</label>
                <input type="number" step="0.01" name="cart_add_rate" value={formData.cart_add_rate} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>📆 Days Since Last Order</label>
                <input type="number" name="days_since_last_order" value={formData.days_since_last_order} onChange={handleChange} />
              </div>
              <div className="form-group">
                <label>⭐ Membership Tier (0-3)</label>
                <input type="number" min="0" max="3" name="membership_encoded" value={formData.membership_encoded} onChange={handleChange} />
              </div>
            </div>
            <button type="submit" className="btn-predict" disabled={predicting}>
              {predicting ? (
                <>
                  <Activity size={20} style={{display: 'inline', marginRight: '8px', animation: 'spin 1s linear infinite'}} />
                  Analyzing Customer Profile...
                </>
              ) : (
                <>
                  <Zap size={20} style={{display: 'inline', marginRight: '8px'}} />
                  Generate AI Prediction
                </>
              )}
            </button>
          </form>
        </div>

        <div className="prediction-result">
          {prediction ? (
            <>
              <h3 style={{color: '#cbd5e1', marginBottom: '1rem', fontSize: '1.1rem'}}>
                🎯 Purchase Probability Score
              </h3>
              <div className={prediction.purchase_probability >= 0.5 ? 'prob-circle high-prob' : 'prob-circle low-prob'}>
                {(prediction.purchase_probability * 100).toFixed(1)}%
              </div>
              <p style={{
                fontSize: '1.3rem', 
                fontWeight: '600',
                color: prediction.prediction === 1 ? '#10b981' : '#ef4444',
                marginTop: '1rem'
              }}>
                {prediction.prediction === 1 ? '✅ High Purchase Intent' : '⚠️ Low Purchase Intent'}
              </p>
              <div style={{
                marginTop: '2rem',
                padding: '1.5rem',
                background: prediction.prediction === 1 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                borderRadius: '12px',
                textAlign: 'left',
                width: '100%'
              }}>
                <h4 style={{color: '#cbd5e1', marginBottom: '1rem', fontSize: '0.95rem'}}>
                  💡 AI Recommendations
                </h4>
                <ul style={{color: '#94a3b8', fontSize: '0.875rem', lineHeight: '1.8', paddingLeft: '1.5rem'}}>
                  {prediction.prediction === 1 ? (
                    <>
                      <li>Send personalized product recommendations</li>
                      <li>Offer exclusive loyalty rewards</li>
                      <li>Enable priority customer support</li>
                    </>
                  ) : (
                    <>
                      <li>Launch re-engagement email campaign</li>
                      <li>Provide limited-time discount offers</li>
                      <li>Analyze browsing patterns for insights</li>
                    </>
                  )}
                </ul>
              </div>
            </>
          ) : (
            <div style={{color: '#94a3b8', opacity: 0.6, textAlign: 'center'}}>
              <Target size={64} style={{marginBottom: '1.5rem', opacity: 0.5}} />
              <h3 style={{fontSize: '1.2rem', marginBottom: '0.5rem'}}>Ready for Prediction</h3>
              <p style={{fontSize: '0.95rem'}}>
                Enter customer profile data and click the button to generate real-time ML predictions
              </p>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default App;
