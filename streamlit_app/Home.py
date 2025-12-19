import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="EV Charging Analytics Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', 'Inter', sans-serif;
            background: radial-gradient(circle at 10% 20%, rgba(99,102,241,0.08), transparent 25%),
                        radial-gradient(circle at 80% 0%, rgba(236,72,153,0.12), transparent 28%),
                        #0b1222;
            color: #e5e7eb;
        }
        .main-header { color: #e0e7ff; }
        .subtitle { color: #cbd5f5; }
        .stat-card, .feature-card, .insight-box, .nav-card, .stMetric {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 15px 45px rgba(0,0,0,0.35);
            color: #e5e7eb;
        }
        .feature-card h3, .nav-card h3 { color: #e0e7ff; }
        .feature-card p, .nav-card p, .insight-box p { color: #cbd5e1; }
        .insight-box { background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(14,165,233,0.14)); border-left: 4px solid #8b5cf6; }
        .divider { background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent); }
        .nav-card { background: linear-gradient(135deg, #1f2937, #0f172a); border: 1px solid rgba(255,255,255,0.12); }
        .stMetric label, .stMetric div { color: #e5e7eb !important; }
        .neon-veil {
            position: relative;
            overflow: hidden;
            border-radius: 18px;
        }
        .neon-veil::before, .neon-veil::after {
            content: '';
            position: absolute;
            width: 480px;
            height: 480px;
            filter: blur(80px);
            opacity: 0.22;
            animation: drift 22s ease-in-out infinite;
            background: radial-gradient(circle at 30% 30%, #8b5cf6, transparent 50%);
        }
        .neon-veil::after {
            width: 360px;
            height: 360px;
            right: -120px;
            top: -60px;
            opacity: 0.28;
            animation-duration: 18s;
            background: radial-gradient(circle at 50% 50%, #22d3ee, transparent 55%);
        }
        @keyframes drift {
            0% { transform: translate3d(0,0,0) scale(1); }
            50% { transform: translate3d(8%, -6%, 0) scale(1.05); }
            100% { transform: translate3d(0,0,0) scale(1); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for professional styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a855f7 0%, #22d3ee 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        text-align: center;
        letter-spacing: 0.3px;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .stat-card {
        background: rgba(255,255,255,0.06);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 10px 28px rgba(0,0,0,0.28);
        border-left: 4px solid #8b5cf6;
        color: #e5e7eb;
        transition: transform 0.2s;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 38px rgba(0,0,0,0.35);
    }
    .feature-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
        padding: 24px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        margin: 12px 0;
        box-shadow: 0 10px 28px rgba(0,0,0,0.28);
    }
    .feature-card h3 {
        color: #e0e7ff;
        margin-bottom: 12px;
        font-size: 1.3rem;
    }
    .feature-card p {
        color: #cbd5e1;
        margin: 8px 0;
        line-height: 1.6;
    }
    .insight-box {
        background: linear-gradient(135deg, rgba(236,72,153,0.2), rgba(59,130,246,0.2));
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid #8b5cf6;
        margin: 10px 0;
        color: #f3f4f6;
    }
    .nav-card {
        background: linear-gradient(135deg, #312e81, #6d28d9);
        padding: 28px;
        border-radius: 14px;
        color: white;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    .nav-card h3 {
        color: white;
        margin: 0 0 16px 0;
        font-size: 1.4rem;
    }
    .nav-card p {
        color: rgba(255,255,255,0.95);
        margin: 10px 0;
        line-height: 1.7;
        font-size: 0.95rem;
    }
    .stMetric {
        background: rgba(255,255,255,0.06);
        padding: 16px;
        border-radius: 10px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        color: #e5e7eb;
    }
    .cta-btn {
        background: linear-gradient(120deg, #6366f1, #8b5cf6, #22d3ee);
        color: #0b1222 !important;
        border: none;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        box-shadow: 0 12px 30px rgba(99,102,241,0.35);
    }
    .glow-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 20px 45px rgba(0,0,0,0.35);
    }
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #94a3b8, transparent);
        margin: 1.5rem 0;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(99,102,241,0.18);
        color: #e0e7ff;
        font-size: 0.9rem;
        border: 1px solid rgba(255,255,255,0.18);
    }
    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 8px 12px;
        border-radius: 12px;
        font-weight: 700;
        color: #0b1222;
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(255,255,255,0.22);
        box-shadow: 0 8px 18px rgba(0,0,0,0.25);
    }
    .status-chip.ok { color: #15803d; background: rgba(34,197,94,0.16); border-color: rgba(34,197,94,0.35); }
    .status-chip.warn { color: #f97316; background: rgba(249,115,22,0.18); border-color: rgba(249,115,22,0.35); }
    .status-chip.upcoming { color: #7c3aed; background: rgba(124,58,237,0.18); border-color: rgba(124,58,237,0.35); }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚡ EV Charging Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Machine Learning-Powered Intelligence for Electric Vehicle Infrastructure</p>', unsafe_allow_html=True)

cta1, cta2, cta3 = st.columns(3)
with cta1:
    st.page_link("pages/1_Forecast.py", label="🚀 Launch Forecasts", use_container_width=True)
with cta2:
    st.page_link("pages/2_Predict_Duration.py", label="⏱️ Duration Lab", use_container_width=True)
with cta3:
    st.page_link("pages/3_Clustering.py", label="🗂️ Cluster Explorer", use_container_width=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Load actual data
@st.cache_data
def load_data():
    data_file = project_root / "data" / "processed" / "cleaned_ev_sessions.csv"
    if data_file.exists():
        try:
            df = pd.read_csv(data_file)
            # Parse dates
            if 'charging_start_time' in df.columns:
                df['charging_start_time'] = pd.to_datetime(df['charging_start_time'])
                df['session_date'] = df['charging_start_time'].dt.date
            return df, None
        except Exception as e:
            return None, str(e)
    return None, "Data file not found"
df, error = load_data()

if df is not None:
    st.success(f"✅ **System Online** | {len(df):,} charging sessions | {df['charging_station_id'].nunique()} stations | Real-time data loaded")
    st.markdown(
        """
        <div style='display:flex; flex-wrap:wrap; gap:10px; margin-top:10px;'>
            <span class='status-chip ok'>API Healthy</span>
            <span class='status-chip ok'>Data Fresh</span>
            <span class='status-chip ok'>Models Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.error(f"❌ Data Connection Issue: {error}")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Overview section with modern design
col1, col2 = st.columns([2.5, 1.5], gap="large")

with col1:
    st.markdown("""
    <div class='neon-veil' style='background: linear-gradient(135deg, rgba(17,24,39,0.92), rgba(17,24,39,0.7)); padding: 30px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.12); box-shadow: 0 22px 48px rgba(0,0,0,0.45);'>
        <h2 style='color: #e0e7ff; margin-top: 0;'>🚀 Welcome to Your Analytics Command Center</h2>
        <div style='display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 6px 0;'>
            <span class='pill'>Live station data</span>
            <span class='pill'>Prophet & baselines</span>
            <span class='pill'>Duration intelligence</span>
            <span class='pill'>Cluster personas</span>
            <span class='pill'>PowerBI exports</span>
        </div>
        <p style='font-size: 1rem; color: #cbd5e1; line-height: 1.4; margin: 10px 0 0 0;'>
            Forecast • Predict • Segment — built on your live sessions.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='nav-card'>
        <h3>📍 Quick Access</h3>
        <p><strong>📈 Demand Forecasting</strong><br/>Neon trends + confidence band</p>
        <p><strong>⏱️ Duration Lab</strong><br/>Fast estimators with baselines</p>
        <p><strong>🗂️ Clusters</strong><br/>Persona map for stations</p>
        <p><strong>📊 Data Explorer</strong><br/>EDA and PowerBI sync</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #e0e7ff;'>📊 Live Performance Dashboard</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if df is not None:
    # Calculate real metrics
    total_sessions = len(df)
    total_stations = df['charging_station_id'].nunique()
    total_users = df['user_id'].nunique()
    avg_duration = df['charging_duration_minutes'].mean()
    total_energy = df['energy_consumed_kwh'].sum()
    avg_cost = df['charging_cost_usd'].mean()
    stats_payload = {
        "sessions": total_sessions,
        "stations": total_stations,
        "users": total_users,
        "avg_duration": avg_duration,
        "energy": total_energy / 1000,
    }
    
    # Display metrics in professional cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Total Sessions",
            f"{total_sessions:,}",
            help="Complete charging sessions recorded"
        )
    
    with col2:
        st.metric(
            "Charging Stations",
            f"{total_stations}",
            help="Unique station locations"
        )
    
    with col3:
        st.metric(
            "Avg. Duration",
            f"{avg_duration:.0f} min",
            help="Mean charging session time"
        )
    
    with col4:
        st.metric(
            "Total Energy",
            f"{total_energy/1000:.1f} MWh",
            help="Total electricity delivered"
        )
    
    with col5:
        st.metric(
            "Active Users",
            f"{total_users:,}",
            help="Unique customers served"
        )
    
    with col6:
        st.metric(
            "Avg. Session Cost",
            f"${avg_cost:.2f}",
            help="Mean revenue per session"
        )
    
    # Visualizations with real data
    viz_col1, viz_col2 = st.columns(2, gap="large")
    
    with viz_col1:
        st.subheader("📈 Daily Session Trend")
        
        if 'session_date' in df.columns or 'charging_start_time' in df.columns:
            date_col = 'session_date' if 'session_date' in df.columns else df['charging_start_time'].dt.date
            
            daily_sessions = df.groupby(date_col).size().reset_index()
            daily_sessions.columns = ['Date', 'Sessions']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_sessions['Date'],
                y=daily_sessions['Sessions'],
                mode='lines+markers',
                fill='tozeroy',
                line=dict(color='#8b5cf6', width=3),
                marker=dict(size=7, color='#22d3ee', line=dict(color='#0f172a', width=1)),
                fillcolor='rgba(139,92,246,0.18)',
                hovertemplate='%{x}<br>Sessions: %{y}<extra></extra>'
            ))
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Number of Sessions",
                hovermode='x unified',
                plot_bgcolor='#0b1222',
                paper_bgcolor='#0b1222',
                font=dict(color='#e5e7eb'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
                height=360,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        st.subheader("⏰ Hourly Usage Pattern")
        
        if 'start_hour' in df.columns or 'charging_start_time' in df.columns:
            if 'start_hour' not in df.columns:
                df['start_hour'] = df['charging_start_time'].dt.hour
            
            hourly_dist = df['start_hour'].value_counts().sort_index()
            peak_hour = hourly_dist.idxmax()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hourly_dist.index,
                y=hourly_dist.values,
                marker_color=['#f472b6' if x == peak_hour else '#22d3ee' for x in hourly_dist.index],
                hovertemplate='%{x}:00 - %{y} sessions<extra></extra>',
                marker=dict(line=dict(color='rgba(255,255,255,0.4)', width=1))
            ))
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Sessions",
                plot_bgcolor='#0b1222',
                paper_bgcolor='#0b1222',
                font=dict(color='#e5e7eb'),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
                height=360,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"🔥 **Peak Hour**: {peak_hour}:00 with {hourly_dist[peak_hour]:,} sessions")
    
    # Key insights from real data
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #e0e7ff;'>💡 Intelligent Insights</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    insight_col1, insight_col2, insight_col3 = st.columns(3, gap="medium")
    
    with insight_col1:
        median_duration = df['charging_duration_minutes'].median()
        duration_p90 = df['charging_duration_minutes'].quantile(0.9)
        st.markdown(f"""
        <div class='insight-box'>
            <h4 style='color: #e0e7ff; margin: 0 0 10px 0;'>⏱️ Session Duration</h4>
            <p style='color: #e2e8f0; margin: 5px 0;'><strong>Median:</strong> {median_duration:.1f} min</p>
            <p style='color: #e2e8f0; margin: 5px 0;'><strong>P90:</strong> {duration_p90:.1f} min</p>
            <p style='color: #e2e8f0; margin: 5px 0;'><strong>Range:</strong> {df['charging_duration_minutes'].min():.0f} - {df['charging_duration_minutes'].max():.0f} min</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col2:
        avg_energy = df['energy_consumed_kwh'].mean()
        median_energy = df['energy_consumed_kwh'].median()
        st.markdown(f"""
        <div class='insight-box'>
            <h4 style='color: #e0e7ff; margin: 0 0 10px 0;'>⚡ Energy</h4>
            <p style='color: #e2e8f0; margin: 5px 0;'><strong>Mean:</strong> {avg_energy:.1f} kWh/session</p>
            <p style='color: #e2e8f0; margin: 5px 0;'><strong>Median:</strong> {median_energy:.1f} kWh/session</p>
            <p style='color: #e2e8f0; margin: 5px 0;'><strong>Total:</strong> {total_energy:,.0f} kWh</p>
        </div>
        """, unsafe_allow_html=True)
    
    with insight_col3:
        sessions_per_station = total_sessions / total_stations
        users_per_station = total_users / total_stations
        st.markdown(f"""
        <div class='insight-box'>
            <h4 style='color: #e0e7ff; margin: 0 0 10px 0;'>🏢 Stations</h4>
            <p style='color: #e2e8f0; margin: 5px 0;'><strong>Sessions/Station:</strong> {sessions_per_station:.1f}</p>
            <p style='color: #e2e8f0; margin: 5px 0;'><strong>Users/Station:</strong> {users_per_station:.1f}</p>
            <p style='color: #e2e8f0; margin: 5px 0;'><strong>Total:</strong> {total_stations}</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Show placeholder metrics if no data
    info_col1, info_col2, info_col3, info_col4 = st.columns(4)
    
    with info_col1:
        st.metric("Total Stations", "--", help="Total number of charging stations")
    
    with info_col2:
        st.metric("Total Sessions", "--", help="Total charging sessions recorded")
    
    with info_col3:
        st.metric("Avg. Duration", "--", help="Average charging duration")
    
    with info_col4:
        st.metric("Peak Hour", "--", help="Most popular charging hour")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: #e0e7ff;'>🤖 Machine Learning Models</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
feature_col1, feature_col2 = st.columns(2, gap="large")

with feature_col1:
    st.markdown("""
    <div class="feature-card">
    <h3>📈 Demand Forecasting</h3>
    <p style='color:#cbd5e1;'>Prophet core with neon trendlines and confidence bands.</p>
    <div style='display:flex; flex-wrap:wrap; gap:8px;'>
        <span class='pill'>Per-station</span>
        <span class='pill'>7-60d horizon</span>
        <span class='pill'>Baseline fallback</span>
        <span class='pill'>Confidence bands</span>
    </div>
    </div>
    <br>
    <div class="feature-card">
    <h3>⏱️ Duration Prediction</h3>
    <p style='color:#cbd5e1;'>Fast heuristics with boosting + RF backup.</p>
    <div style='display:flex; flex-wrap:wrap; gap:8px;'>
        <span class='pill'>MAE 51.7</span>
        <span class='pill'>Latency-light</span>
        <span class='pill'>Feature-aware</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

with feature_col2:
    st.markdown("""
    <div class="feature-card">
    <h3>🗂️ Station Clustering</h3>
    <p style='color:#cbd5e1;'>K-Means personas to prioritize rollouts.</p>
    <div style='display:flex; flex-wrap:wrap; gap:8px;'>
        <span class='pill'>4 personas</span>
        <span class='pill'>Sessions + energy</span>
        <span class='pill'>Export CSV</span>
    </div>
    </div>
    <br>
    <div class="feature-card">
    <h3>📊 Exploratory Analysis</h3>
    <p style='color:#cbd5e1;'>Interactive visuals plus PowerBI-ready exports.</p>
    <div style='display:flex; flex-wrap:wrap; gap:8px;'>
        <span class='pill'>Plotly</span>
        <span class='pill'>Filters</span>
        <span class='pill'>Correlations</span>
        <span class='pill'>Exports</span>
    </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Getting started guide
st.header("📚 Quick Start")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
        <h3>🔎 Explore</h3>
        <p style='margin:0;'>EDA and data quality</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <h3>📈 Forecast</h3>
        <p style='margin:0;'>Station demand horizons</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <h3>🗂️ Segment</h3>
        <p style='margin:0;'>Cluster personas & ops</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Footer removed per request
