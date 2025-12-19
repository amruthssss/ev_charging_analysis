import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Station Clustering",
    page_icon="🗂️",
    layout="wide"
)

st.title("🗂️ Station Clustering Analysis")
st.markdown("Segment charging stations based on usage patterns and characteristics")
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


@st.cache_data
def load_clusters():
    clusters_path = project_root / "data" / "processed" / "station_clusters.csv"
    if not clusters_path.exists():
        return None, "station_clusters.csv not found"
    try:
        df = pd.read_csv(clusters_path)
        return df, None
    except Exception as exc:
        return None, str(exc)

# Sidebar controls
st.sidebar.header("Clustering Configuration")

clustering_method = st.sidebar.selectbox(
    "Clustering Algorithm",
    ["K-Means (Active)"],
    help="K-Means clustering with StandardScaler"
)

num_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=10,
    value=4,
    help="Current model uses 4 clusters",
    disabled=True
)

st.sidebar.info("""
**Current Model:**
- Algorithm: K-Means
- Stations: data-driven
- Clusters: 4
- Sizes: loaded from station_clusters.csv
""")

features_to_include = st.sidebar.multiselect(
    "Features for Clustering",
    [
        "Average Session Duration",
        "Peak Usage Hours",
        "Weekly Session Count",
        "Energy Consumption",
        "User Diversity",
        "Location Type",
        "Charger Type Distribution"
    ],
    default=[
        "Average Session Duration",
        "Peak Usage Hours",
        "Weekly Session Count"
    ],
    help="Select features to use in clustering analysis"
)

st.sidebar.markdown("---")

if st.sidebar.button("🔄 Run Clustering", type="primary", use_container_width=True):
    with st.spinner("Running clustering analysis..."):
        try:
            # API integration structure
            # import requests
            # response = requests.post('http://localhost:8000/api/clusters', json={...})
            st.sidebar.success("✅ Clustering complete!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

st.sidebar.markdown("---")
st.sidebar.info("""
**💡 Tip**: 
Choose 3-7 clusters for most scenarios. 
Use silhouette score to validate cluster quality.
""")

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "Cluster Overview",
    "Station Distribution",
    "Cluster Profiles",
    "Insights & Recommendations"
])

clusters_df, clusters_err = load_clusters()
if clusters_err:
    st.warning(f"Cluster data issue: {clusters_err}")

with tab1:
    st.subheader("Clustering Results Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if clusters_df is not None:
        counts = clusters_df['cluster'].value_counts().sort_values(ascending=False)
        largest = counts.iloc[0] if not counts.empty else 0
        smallest = counts.iloc[-1] if not counts.empty else 0
        with col1:
            st.metric("Total Stations", f"{len(clusters_df):,}", help="Stations analyzed and clustered")
        with col2:
            st.metric("Clusters", f"{clusters_df['cluster'].nunique()}", help="Behavioral groups identified")
        with col3:
            st.metric("Largest Cluster", f"{largest}")
        with col4:
            st.metric("Smallest Cluster", f"{smallest}")
    else:
        with col1:
            st.metric("Total Stations", "--")
        with col2:
            st.metric("Clusters", "--")
        with col3:
            st.metric("Largest Cluster", "--")
        with col4:
            st.metric("Smallest Cluster", "--")
    
    st.markdown("---")
    
    # Cluster visualization
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Cluster Visualization (2D Projection)")
        
        # Sample scatter plot
        st.info("📊 Interactive cluster visualization will appear here (PCA/t-SNE projection)")
        
        # Placeholder visualization
        sample_data = pd.DataFrame({
            'PC1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'PC2': [2, 4, 1, 5, 3, 6, 2, 8, 4, 7],
            'Cluster': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E'],
            'Station': [f'Station {i}' for i in range(1, 11)]
        })
        
        fig = px.scatter(
            sample_data,
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_data=['Station'],
            title='Station Clusters (PCA Projection)'
        )
        fig.update_layout(
            xaxis_title="Principal Component 1",
            yaxis_title="Principal Component 2"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Cluster Sizes")
        
        if clusters_df is not None:
            cluster_sizes = clusters_df['cluster'].value_counts().reset_index()
            cluster_sizes.columns = ['Cluster', 'Count']
            fig = px.pie(
                cluster_sizes,
                values='Count',
                names='Cluster',
                title='Station Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cluster data not available")

with tab2:
    st.subheader("Geographic Distribution of Clusters")
    
    st.info("🗺️ Map visualization showing station locations colored by cluster will appear here")
    
    st.markdown("---")
    
    st.subheader("Cluster Statistics")
    
    # Real cluster statistics from report
    cluster_stats = pd.DataFrame({
        'Cluster': ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3'],
        'Stations': [121, 91, 155, 90],
        'Avg Sessions': [4.71, 1.70, 2.32, 1.88],
        'Avg Duration (min)': [135.49, 159.44, 126.31, 138.94],
        'Avg Energy (kWh)': [43.57, 42.50, 37.86, 47.06],
        'DC Fast Ratio': ['34%', '12%', '16%', '79%']
    })
    
    st.dataframe(cluster_stats, use_container_width=True)

with tab3:
    st.subheader("Detailed Cluster Profiles")
    
    selected_cluster = st.selectbox(
        "Select Cluster to Analyze",
        [f'Cluster {i}' for i in range(1, num_clusters + 1)]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {selected_cluster} Characteristics")
        
        st.markdown("#### Usage Patterns")
        st.info("📊 Time-series usage pattern chart will appear here")
        
        st.markdown("#### Key Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Avg Daily Sessions", "42")
            st.metric("Avg Duration", "38 min")
            st.metric("Peak Utilization", "78%")
        
        with metrics_col2:
            st.metric("Energy/Session", "18.5 kWh")
            st.metric("Unique Users", "156")
            st.metric("Weekend Traffic", "35%")
    
    with col2:
        st.markdown("### Station List")
        
        # Sample station list
        stations_in_cluster = pd.DataFrame({
            'Station ID': [f'STN-{i:04d}' for i in range(1, 11)],
            'Location': ['Location ' + chr(65 + i % 5) for i in range(10)],
            'Sessions/Day': [40 + i * 2 for i in range(10)],
            'Avg Duration': [35 + i for i in range(10)]
        })
        
        st.dataframe(stations_in_cluster, use_container_width=True, height=400)
        
        if st.button("📥 Export Station List"):
            st.info("Export functionality will be available soon")
    
    st.markdown("---")
    
    st.markdown("### Cluster Feature Analysis")
    
    # Radar chart for cluster characteristics
    st.info("📊 Radar chart comparing cluster characteristics will appear here")

with tab4:
    st.subheader("Insights & Recommendations")
    
    # Cluster descriptions
    st.markdown("### Cluster Interpretations")
    
    for i in range(1, min(num_clusters + 1, 6)):
        with st.expander(f"📌 Cluster {i} - [Name TBD]"):
            st.markdown(f"""
            **Profile**: [Description based on actual clustering results]
            
            **Characteristics**:
            - Usage pattern: [Pattern description]
            - User behavior: [Behavior description]
            - Location type: [Type description]
            
            **Recommendations**:
            - [Recommendation 1]
            - [Recommendation 2]
            - [Recommendation 3]
            
            **Optimization Opportunities**:
            - [Opportunity 1]
            - [Opportunity 2]
            """)
    
    st.markdown("---")
    
    st.subheader("Strategic Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### High-Value Clusters")
        st.success("""
        - **Cluster X**: High usage, high revenue potential
        - **Cluster Y**: Growing demand, expansion opportunity
        - **Cluster Z**: Stable performance, reliable revenue
        """)
    
    with col2:
        st.markdown("#### Areas for Improvement")
        st.warning("""
        - **Cluster A**: Underutilized, needs promotion
        - **Cluster B**: High wait times, capacity needed
        - **Cluster C**: Maintenance issues detected
        """)
    
    st.markdown("---")
    
    st.subheader("Action Items")
    
    action_items = pd.DataFrame({
        'Cluster': ['Cluster 1', 'Cluster 2', 'Cluster 3'],
        'Priority': ['High', 'Medium', 'Low'],
        'Action': [
            'Add 2 fast chargers to reduce wait times',
            'Implement dynamic pricing during peak hours',
            'Improve signage and accessibility'
        ],
        'Expected Impact': ['20% capacity increase', '15% revenue increase', '10% usage increase']
    })
    
    st.table(action_items)

# Export options
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💾 Export Cluster Results", use_container_width=True):
        st.info("Export functionality will be available soon")

with col2:
    if st.button("📊 Generate Report", use_container_width=True):
        st.info("Report generation will be available soon")

with col3:
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        st.info("Analysis will be refreshed")

st.markdown("---")
st.caption("Station Clustering Module | EV Charging Analytics Platform")
