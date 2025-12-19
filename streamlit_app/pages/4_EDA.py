import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import plotly.io as pio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis")
st.markdown("Deep dive into EV charging patterns and trends")
st.markdown("---")

# Plotly dark defaults for consistent theme
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = ["#a5b4fc", "#f472b6", "#22d3ee", "#fbbf24"]
pio.templates.default = "plotly_dark"

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
        .stMetric label, .stMetric div { color: #e5e7eb !important; }
        h1, h2, h3, h4 { color: #e0e7ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar - Data filters
st.sidebar.header("Data Filters")

# Try to load actual data
data_file = project_root / "data" / "processed" / "cleaned_ev_sessions.csv"

if data_file.exists():
    try:
        df = pd.read_csv(data_file)
        df["charging_start_time"] = pd.to_datetime(df["charging_start_time"])
        df["session_date"] = df["charging_start_time"].dt.date
        st.sidebar.success(f"✅ Loaded {len(df):,} records")
        
        # Date range filter
        min_date = df["charging_start_time"].min().date()
        max_date = df["charging_start_time"].max().date()
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            mask = (df["charging_start_time"].dt.date >= start) & (df["charging_start_time"].dt.date <= end)
            df = df[mask]
        
        # Station filter
        all_stations = df['charging_station_id'].unique()
        selected_stations = st.sidebar.multiselect(
            "Stations",
            options=['All'] + list(all_stations),
            default=['All']
        )
        
        if 'All' not in selected_stations and selected_stations:
            df = df[df['charging_station_id'].isin(selected_stations)]
        
        data_loaded = True
    except Exception as e:
        st.sidebar.error(f"Error loading data: {e}")
        data_loaded = False
        df = None
else:
    st.sidebar.warning("⚠️ Data file not found")
    data_loaded = False
    df = None

st.sidebar.markdown("---")

# Refresh button
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.rerun()

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Temporal Analysis",
    "Station Analysis",
    "Usage Patterns",
    "Statistical Summary"
])

with tab1:
    st.subheader("Dataset Overview")
    
    if data_loaded and df is not None:
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Sessions", f"{len(df):,}")
        
        with col2:
            st.metric("Unique Stations", f"{df['charging_station_id'].nunique():,}")
        
        with col3:
            st.metric("Avg Duration", f"{df['charging_duration_minutes'].mean():.1f} min")
        
        with col4:
            st.metric("Total Energy", f"{df['energy_consumed_kwh'].sum():.1f} kWh")
        
        with col5:
            st.metric("Unique Users", f"{df['user_id'].nunique():,}")
        
        st.markdown("---")
        
        # Data sample
        st.subheader("Data Sample")
        st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("---")
        
        # Column information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
        with col2:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Type': df.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, use_container_width=True, height=300)
    
    else:
        st.info("📁 No data loaded. Please ensure the data file exists at: `data/processed/cleaned_ev_sessions.csv`")

with tab2:
    st.subheader("Temporal Patterns")
    
    if data_loaded and df is not None:
        # Daily trends
        st.markdown("### Daily Trends")
        
        daily_counts = df.groupby(df["charging_start_time"].dt.date).size().reset_index()
        daily_counts.columns = ['Date', 'Sessions']
        
        fig = px.line(
            daily_counts,
            x='Date',
            y='Sessions',
            title='Daily Session Count'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Hourly Distribution")
            
            hourly_dist = df['start_hour'].value_counts().sort_index()
            
            fig = px.bar(
                x=hourly_dist.index,
                y=hourly_dist.values,
                labels={'x': 'Hour of Day', 'y': 'Number of Sessions'},
                title='Sessions by Hour of Day'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Day of Week Distribution")
            
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_dist = df['day_of_week'].value_counts()
            dow_dist = dow_dist.reindex([d for d in dow_order if d in dow_dist.index])
            
            fig = px.bar(
                x=dow_dist.index,
                y=dow_dist.values,
                labels={'x': 'Day of Week', 'y': 'Number of Sessions'},
                title='Sessions by Day of Week'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Heatmap
        st.markdown("### Usage Heatmap (Hour × Day of Week)")
        st.info("📊 Heatmap showing session density across hours and days will appear here")
    
    else:
        st.info("📁 Load data to view temporal analysis")

with tab3:
    st.subheader("Station-Level Analysis")
    
    if data_loaded and df is not None:
        # Top stations
        st.markdown("### Top Stations by Session Count")
        
        top_stations = df['charging_station_id'].value_counts().head(10)
        
        fig = px.bar(
            x=top_stations.values,
            y=top_stations.index.astype(str),
            orientation='h',
            labels={'x': 'Number of Sessions', 'y': 'Station ID'},
            title='Top 10 Most Active Stations'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Station Statistics")
            
            station_stats = df.groupby('charging_station_id').agg({
                'user_id': 'count',
                'charging_duration_minutes': 'mean',
                'energy_consumed_kwh': 'sum'
            }).round(2)
            station_stats.columns = ['Sessions', 'Avg Duration (min)', 'Total Energy (kWh)']
            
            st.dataframe(station_stats.head(15), use_container_width=True)
        
        with col2:
            st.markdown("### Station Utilization Distribution")
            
            station_counts = df['charging_station_id'].value_counts()
            
            fig = px.histogram(
                station_counts,
                nbins=30,
                labels={'value': 'Sessions per Station', 'count': 'Number of Stations'},
                title='Distribution of Station Usage'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📁 Load data with station information to view analysis")

with tab4:
    st.subheader("Usage Patterns & Behavior")
    
    if data_loaded and df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Duration Distribution")
            
            fig = px.histogram(
                df,
                x='charging_duration_minutes',
                nbins=50,
                labels={'charging_duration_minutes': 'Duration (minutes)'},
                title='Charging Duration Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Duration Statistics:**")
            st.write(df['charging_duration_minutes'].describe())
        
        with col2:
            st.markdown("### Energy Consumption Distribution")
            
            fig = px.histogram(
                df,
                x='energy_consumed_kwh',
                nbins=50,
                labels={'energy_consumed_kwh': 'Energy (kWh)'},
                title='Energy Consumption Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("**Energy Statistics:**")
            st.write(df['energy_consumed_kwh'].describe())
        
        st.markdown("---")
        
        # Correlation analysis
        st.markdown("### Feature Correlations")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                title='Correlation Matrix',
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📁 Load data to view usage patterns")

with tab5:
    st.subheader("Statistical Summary")
    
    if data_loaded and df is not None:
        # Numeric summary
        st.markdown("### Numeric Features Summary")
        numeric_summary = df.describe()
        st.dataframe(numeric_summary, use_container_width=True)
        
        st.markdown("---")
        
        # Categorical summary
        st.markdown("### Categorical Features Summary")
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            for col in categorical_cols[:5]:  # Show first 5 categorical columns
                with st.expander(f"📋 {col}"):
                    value_counts = df[col].value_counts().head(10)
                    st.write(f"**Unique Values:** {df[col].nunique()}")
                    st.write(f"**Most Common:**")
                    st.dataframe(value_counts)
        
        st.markdown("---")
        
        # Missing values
        st.markdown("### Missing Values Analysis")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")
    
    else:
        st.info("📁 Load data to view statistical summary")

# Export section
st.markdown("---")
st.subheader("📥 Export & Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if data_loaded and df is not None:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Export Filtered Data",
            data=csv,
            file_name=f"ev_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.button("💾 Export Data (No Data)", disabled=True, use_container_width=True)

with col2:
    if st.button("📊 Generate PDF Report", use_container_width=True):
        st.info("📄 PDF report generation coming soon")

with col3:
    if st.button("📈 Export Charts", use_container_width=True):
        st.info("🖼️ Chart export feature in development")

with col4:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.markdown("---")
st.caption("Exploratory Data Analysis Module | EV Charging Analytics Platform")
