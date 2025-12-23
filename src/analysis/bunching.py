"""
Bus bunching detection and analysis.
Bunching occurs when buses cluster together, causing uneven headways and increased passenger wait times.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def detect_bunching(
    vehicle_positions: pd.DataFrame,
    route_id: str,
    direction_id: Optional[int] = None,
    bunching_threshold_km: float = 0.5
) -> pd.DataFrame:
    """
    Detect bus bunching on a route.
    
    Bunching occurs when multiple buses are within threshold distance of each other.
    
    Args:
        vehicle_positions: DataFrame with vehicle_id, route_id, latitude, longitude, timestamp
        route_id: Route to analyze
        direction_id: Optional direction filter
        bunching_threshold_km: Distance threshold for bunching (default 500m)
        
    Returns:
        DataFrame with bunching incidents: timestamp, location, n_buses, severity
    """
    route_vehicles = vehicle_positions[
        vehicle_positions['route_id'] == route_id
    ].copy()
    
    if direction_id is not None:
        route_vehicles = route_vehicles[
            route_vehicles.get('direction_id', 0) == direction_id
        ]
    
    if len(route_vehicles) < 2:
        return pd.DataFrame()
    
    from scipy.spatial.distance import cdist
    
    incidents = []
    
    for timestamp in route_vehicles['timestamp'].unique():
        snapshot = route_vehicles[route_vehicles['timestamp'] == timestamp]
        
        if len(snapshot) < 2:
            continue
        
        positions = snapshot[['latitude', 'longitude']].values
        
        distances_km = cdist(positions, positions, metric='euclidean') * 111
        
        np.fill_diagonal(distances_km, np.inf)
        
        min_distances = distances_km.min(axis=1)
        
        bunched = snapshot[min_distances <= bunching_threshold_km].copy()
        
        if len(bunched) >= 2:
            n_buses = len(bunched)
            avg_distance = distances_km[
                distances_km <= bunching_threshold_km
            ].mean()
            
            severity = 'high' if n_buses >= 3 else 'medium'
            
            incidents.append({
                'timestamp': timestamp,
                'route_id': route_id,
                'n_buses_bunched': n_buses,
                'avg_distance_km': avg_distance,
                'severity': severity,
                'center_lat': bunched['latitude'].mean(),
                'center_lon': bunched['longitude'].mean()
            })
    
    return pd.DataFrame(incidents)


def analyze_headway_variance(
    delays_df: pd.DataFrame,
    route_id: str,
    stop_id: Optional[str] = None
) -> Dict[str, float]:
    """
    Analyze headway variance as indicator of bunching.
    
    High variance indicates uneven service (bunching).
    
    Args:
        delays_df: DataFrame with route_id, stop_id, actual_local_time, computed_delay_sec
        route_id: Route to analyze
        stop_id: Optional specific stop
        
    Returns:
        Dictionary with variance metrics
    """
    route_data = delays_df[delays_df['route_id'] == route_id].copy()
    
    if stop_id:
        route_data = route_data[route_data['stop_id'] == stop_id]
    
    if len(route_data) < 2:
        return {}
    
    route_data = route_data.sort_values('actual_local_time')
    
    route_data['headway_sec'] = route_data['actual_local_time'].diff().dt.total_seconds()
    route_data = route_data[route_data['headway_sec'].notna()]
    
    if len(route_data) == 0:
        return {}
    
    headways = route_data['headway_sec'].values
    
    mean_headway = headways.mean()
    std_headway = headways.std()
    cv = std_headway / mean_headway if mean_headway > 0 else 0
    
    return {
        'mean_headway_min': mean_headway / 60,
        'std_headway_min': std_headway / 60,
        'coefficient_of_variation': cv,
        'min_headway_min': headways.min() / 60,
        'max_headway_min': headways.max() / 60,
        'bunching_score': cv  # Higher CV = more bunching
    }


def estimate_passenger_wait_time_impact(
    delays_df: pd.DataFrame,
    route_id: str,
    scheduled_headway_min: float
) -> Dict[str, float]:
    """
    Estimate passenger wait time impact from bunching/irregular headways.
    
    Irregular headways increase average wait time beyond scheduled_headway/2.
    
    Args:
        delays_df: DataFrame with route_id, stop_id, actual_local_time
        route_id: Route to analyze
        scheduled_headway_min: Scheduled headway in minutes
        
    Returns:
        Dictionary with wait time metrics
    """
    route_data = delays_df[delays_df['route_id'] == route_id].copy()
    route_data = route_data.sort_values('actual_local_time')
    
    route_data['headway_sec'] = route_data['actual_local_time'].diff().dt.total_seconds()
    route_data = route_data[route_data['headway_sec'].notna()]
    
    if len(route_data) == 0:
        return {}
    
    headways_min = route_data['headway_sec'].values / 60
    
    expected_wait_min = scheduled_headway_min / 2
    actual_wait_min = (headways_min ** 2).mean() / (2 * headways_min.mean())
    
    wait_time_penalty = actual_wait_min - expected_wait_min
    
    return {
        'expected_wait_min': expected_wait_min,
        'actual_wait_min': actual_wait_min,
        'wait_time_penalty_min': wait_time_penalty,
        'penalty_pct': (wait_time_penalty / expected_wait_min) * 100 if expected_wait_min > 0 else 0
    }


def identify_bunching_hotspots(
    historical_delays: pd.DataFrame,
    min_incidents: int = 5
) -> pd.DataFrame:
    """
    Identify routes/stops with frequent bunching incidents.
    
    Args:
        historical_delays: Historical delay data
        min_incidents: Minimum incidents to be considered hotspot
        
    Returns:
        DataFrame of hotspots sorted by frequency
    """
    route_data = historical_delays.copy()
    route_data = route_data.sort_values('actual_local_time')
    
    route_data['headway_sec'] = route_data.groupby(['route_id', 'stop_id'])['actual_local_time'].diff().dt.total_seconds()
    
    hotspots = []
    
    for (route_id, stop_id), group in route_data.groupby(['route_id', 'stop_id']):
        headways = group['headway_sec'].dropna().values / 60
        
        if len(headways) < min_incidents:
            continue
        
        mean_headway = headways.mean()
        cv = headways.std() / mean_headway if mean_headway > 0 else 0
        
        if cv > 0.5:  # High variance threshold
            hotspots.append({
                'route_id': route_id,
                'stop_id': stop_id,
                'n_observations': len(headways),
                'mean_headway_min': mean_headway,
                'headway_cv': cv,
                'bunching_severity': 'high' if cv > 1.0 else 'medium'
            })
    
    return pd.DataFrame(hotspots).sort_values('headway_cv', ascending=False)

