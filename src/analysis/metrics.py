"""
Route efficiency and performance metrics
"""
import numpy as np
import pandas as pd
from typing import Dict


class RouteMetrics:
    """Calculate operational metrics for routes"""
    
    def __init__(self, gtfs_loader):
        self.loader = gtfs_loader
        
    def route_directness(self, route_id: str) -> float:
        """
        Directness ratio: actual path length / straight-line distance.
        Returns NaN when the route has no usable geometry.
        """
        shapes = self.loader.shapes
        trips = self.loader.trips
        
        route_trips = trips[trips['route_id'] == route_id]
        if route_trips.empty:
            return np.nan
        
        # Prefer trips with a non-null shape_id
        route_trips = route_trips[route_trips['shape_id'].notna()]
        if route_trips.empty:
            return np.nan
        
        trip = route_trips.iloc[0]
        shape_id = trip['shape_id']
        shape = shapes[shapes['shape_id'] == shape_id].sort_values('shape_pt_sequence')
        if shape.empty or len(shape) < 2:
            return np.nan
        
        # Calculate actual path length
        lats = np.radians(shape['shape_pt_lat'].values)
        lons = np.radians(shape['shape_pt_lon'].values)
        
        dlat = np.diff(lats)
        dlon = np.diff(lons)
        a = np.sin(dlat / 2) ** 2 + np.cos(lats[:-1]) * np.cos(lats[1:]) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = 6371 * c  # Earth radius in km
        total_distance = distances.sum()
        
        # Calculate straight-line distance
        lat1, lon1 = lats[0], lons[0]
        lat2, lon2 = lats[-1], lons[-1]
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        straight_distance = 6371 * c
        
        # If the endpoints are essentially the same, directness is undefined
        if straight_distance < 1e-3:
            return np.nan
        
        return float(total_distance / straight_distance)
    
    def stops_per_km(self, route_id: str) -> float:
        """Calculate stop density (stops per kilometer)."""
        trips = self.loader.trips
        stop_times = self.loader.stop_times
        shapes = self.loader.shapes
        
        route_trips = trips[trips['route_id'] == route_id]
        if route_trips.empty:
            return np.nan
        
        route_trips = route_trips[route_trips['shape_id'].notna()]
        if route_trips.empty:
            return np.nan
        
        trip = route_trips.iloc[0]
        trip_stops = stop_times[stop_times['trip_id'] == trip['trip_id']]
        num_stops = len(trip_stops)
        
        # Get route length
        shape_id = trip['shape_id']
        shape = shapes[shapes['shape_id'] == shape_id]
        if shape.empty or 'shape_dist_traveled' not in shape.columns:
            return np.nan
        
        max_dist = shape['shape_dist_traveled'].max()
        if pd.isna(max_dist) or max_dist <= 0:
            return np.nan
        
        # Convert miles to km (GTFS often uses miles)
        distance_km = float(max_dist) * 1.60934
        
        return float(num_stops / distance_km) if distance_km > 0 else np.nan
    
    def trips_per_day(self, route_id: str, service_id: str = None) -> int:
        """Count trips for a route on a given service day"""
        trips = self.loader.trips
        route_trips = trips[trips['route_id'] == route_id]
        
        if service_id:
            route_trips = route_trips[route_trips['service_id'] == service_id]
            
        return len(route_trips)
    
    def service_span_hours(self, route_id: str) -> float:
        """Calculate service span (first to last trip) in hours"""
        trips = self.loader.trips
        stop_times = self.loader.stop_times
        
        route_trips = trips[trips['route_id'] == route_id]['trip_id']
        if route_trips.empty:
            return 0.0
        route_stop_times = stop_times[stop_times['trip_id'].isin(route_trips)]
        
        # Parse time strings (HH:MM:SS)
        def time_to_seconds(t):
            if pd.isna(t):
                return None
            h, m, s = map(int, str(t).split(':'))
            return h * 3600 + m * 60 + s
        
        times = route_stop_times['departure_time'].apply(time_to_seconds).dropna()
        if len(times) == 0:
            return 0.0
        
        span_seconds = times.max() - times.min()
        return float(span_seconds / 3600.0)
    
    def all_routes_summary(self) -> pd.DataFrame:
        """Generate summary metrics for all routes"""
        routes = self.loader.routes
        
        results = []
        for _, route in routes.iterrows():
            route_id = route['route_id']
            try:
                directness = self.route_directness(route_id)
                stops_density = self.stops_per_km(route_id)
                results.append({
                    'route_id': route_id,
                    'route_name': route['route_short_name'],
                    'directness': directness,
                    'stops_per_km': stops_density,
                    'trips_per_day': self.trips_per_day(route_id),
                    'service_span_hours': self.service_span_hours(route_id),
                })
            except Exception:
                pass
                
        return pd.DataFrame(results)

