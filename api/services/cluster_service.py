"""
Cluster Service - Station segmentation and cluster assignment.
"""
from fastapi import HTTPException
from api.utils.loaders import get_station_clusters
from api.schemas import ClusterRequest, ClusterResponse


# Cluster descriptions (customize based on your actual cluster analysis)
CLUSTER_DESCRIPTIONS = {
    0: "Low demand – occasional use station",
    1: "Medium demand – steady weekday station",
    2: "High demand – peak hour dominant station",
    3: "Very high demand – busy urban charging hub"
}


def get_station_cluster(request: ClusterRequest) -> ClusterResponse:
    """
    Get cluster assignment for a station.
    
    Args:
        request: ClusterRequest with station_id
        
    Returns:
        ClusterResponse with cluster number and description
    """
    try:
        # Load station clusters
        clusters_df = get_station_clusters()
        
        # Find the station
        station_row = clusters_df[clusters_df['station_id'] == request.station_id]
        
        if len(station_row) == 0:
            available_stations = clusters_df['station_id'].head(5).tolist()
            raise HTTPException(
                status_code=404,
                detail=f"Station '{request.station_id}' not found. Available stations (sample): {available_stations}"
            )
        
        # Get cluster assignment
        cluster = int(station_row.iloc[0]['cluster'])
        
        # Get description (use default if cluster not in descriptions)
        description = CLUSTER_DESCRIPTIONS.get(
            cluster,
            f"Cluster {cluster} station"
        )
        
        return ClusterResponse(
            station_id=request.station_id,
            cluster=cluster,
            description=description
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cluster lookup failed: {str(e)}"
        )


def get_all_clusters() -> dict:
    """Get summary of all clusters (bonus endpoint)."""
    try:
        clusters_df = get_station_clusters()
        
        cluster_summary = clusters_df.groupby('cluster').agg({
            'station_id': 'count'
        }).reset_index()
        cluster_summary.columns = ['cluster', 'station_count']
        
        return {
            "total_stations": len(clusters_df),
            "total_clusters": int(clusters_df['cluster'].nunique()),
            "clusters": [
                {
                    "cluster": int(row['cluster']),
                    "station_count": int(row['station_count']),
                    "description": CLUSTER_DESCRIPTIONS.get(int(row['cluster']), f"Cluster {int(row['cluster'])}")
                }
                for _, row in cluster_summary.iterrows()
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cluster summary: {str(e)}"
        )
