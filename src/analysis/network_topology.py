"""
Network topology analysis for transit systems.
Graph-based analysis of route connectivity, transfers, and network efficiency.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import networkx as nx


class TransitNetwork:
    """Analyze transit network as a graph"""
    
    def __init__(self, gtfs_loader):
        """
        Args:
            gtfs_loader: GTFSLoader instance with loaded data
        """
        self.loader = gtfs_loader
        self.loader.load_all()
        self.G = None
        self._build_graph()
    
    def _build_graph(self):
        """Build network graph from GTFS data"""
        G = nx.DiGraph()
        
        stops = self.loader.stops
        stop_times = self.loader.stop_times
        trips = self.loader.trips
        
        stop_times = stop_times.merge(
            trips[['trip_id', 'route_id']],
            on='trip_id',
            how='left'
        )
        
        for route_id in stop_times['route_id'].unique():
            route_stops = stop_times[stop_times['route_id'] == route_id].copy()
            route_stops = route_stops.sort_values(['trip_id', 'stop_sequence'])
            
            for trip_id in route_stops['trip_id'].unique():
                trip_stops = route_stops[route_stops['trip_id'] == trip_id]
                trip_stops = trip_stops.sort_values('stop_sequence')
                
                for i in range(len(trip_stops) - 1):
                    from_stop = trip_stops.iloc[i]['stop_id']
                    to_stop = trip_stops.iloc[i + 1]['stop_id']
                    
                    if G.has_edge(from_stop, to_stop):
                        if 'routes' not in G[from_stop][to_stop]:
                            G[from_stop][to_stop]['routes'] = set()
                        G[from_stop][to_stop]['routes'].add(route_id)
                        G[from_stop][to_stop]['weight'] += 1
                    else:
                        G.add_edge(
                            from_stop,
                            to_stop,
                            routes={route_id},
                            weight=1,
                            route_id=route_id
                        )
        
        self.G = G
        return G
    
    def transfer_opportunities(self, max_walk_meters: float = 400) -> pd.DataFrame:
        """
        Identify transfer opportunities between routes.
        
        Args:
            max_walk_meters: Maximum walk distance for transfer (default 400m)
            
        Returns:
            DataFrame of transfer opportunities
        """
        stops = self.loader.stops.copy()
        routes = self.loader.routes
        
        from src.models.optimize import haversine_distance_matrix
        
        stop_coords = stops[['stop_lat', 'stop_lon']].values
        distances_km = haversine_distance_matrix(
            stop_coords[:, 0], stop_coords[:, 1],
            stop_coords[:, 0], stop_coords[:, 1]
        )
        distances = distances_km * 1000
        
        stop_times = self.loader.stop_times
        trips = self.loader.trips
        
        stop_routes = stop_times.merge(
            trips[['trip_id', 'route_id']],
            on='trip_id'
        ).groupby('stop_id')['route_id'].apply(set).to_dict()
        
        transfers = []
        
        for i, stop1 in stops.iterrows():
            routes1 = stop_routes.get(stop1['stop_id'], set())
            if not routes1:
                continue
            
            for j, stop2 in stops.iterrows():
                if i >= j:
                    continue
                
                dist = distances[i, j]
                if dist > max_walk_meters:
                    continue
                
                routes2 = stop_routes.get(stop2['stop_id'], set())
                if not routes2:
                    continue
                
                common_routes = routes1 & routes2
                transfer_routes = (routes1 | routes2) - common_routes
                
                if transfer_routes:
                    transfers.append({
                        'stop1_id': stop1['stop_id'],
                        'stop1_lat': stop1['stop_lat'],
                        'stop1_lon': stop1['stop_lon'],
                        'stop2_id': stop2['stop_id'],
                        'stop2_lat': stop2['stop_lat'],
                        'stop2_lon': stop2['stop_lon'],
                        'distance_m': dist,
                        'routes1': list(routes1),
                        'routes2': list(routes2),
                        'transfer_routes': list(transfer_routes),
                        'n_transfers': len(transfer_routes)
                    })
        
        return pd.DataFrame(transfers)
    
    def route_connectivity(self) -> pd.DataFrame:
        """
        Analyze connectivity between routes via transfers.
        
        Returns:
            DataFrame with connectivity metrics per route pair
        """
        transfers = self.transfer_opportunities()
        
        route_pairs = defaultdict(int)
        
        for _, transfer in transfers.iterrows():
            routes1 = set(transfer['routes1'])
            routes2 = set(transfer['routes2'])
            
            for r1 in routes1:
                for r2 in routes2:
                    if r1 != r2:
                        route_pairs[(r1, r2)] += 1
        
        connectivity = []
        for (r1, r2), count in route_pairs.items():
            connectivity.append({
                'route1': r1,
                'route2': r2,
                'n_transfer_points': count
            })
        
        return pd.DataFrame(connectivity).sort_values('n_transfer_points', ascending=False)
    
    def network_centrality(self) -> pd.DataFrame:
        """
        Calculate network centrality metrics for stops.
        
        Returns:
            DataFrame with centrality scores
        """
        if self.G is None:
            self._build_graph()
        
        stops = self.loader.stops.copy()
        
        betweenness = nx.betweenness_centrality(self.G)
        closeness = nx.closeness_centrality(self.G)
        pagerank = nx.pagerank(self.G)
        
        stops['betweenness_centrality'] = stops['stop_id'].map(betweenness)
        stops['closeness_centrality'] = stops['stop_id'].map(closeness)
        stops['pagerank'] = stops['stop_id'].map(pagerank)
        
        stops = stops.fillna(0)
        
        return stops.sort_values('betweenness_centrality', ascending=False)
    
    def service_overlap(self) -> pd.DataFrame:
        """
        Identify routes with high service overlap.
        
        Returns:
            DataFrame of overlapping route pairs
        """
        stop_times = self.loader.stop_times
        trips = self.loader.trips
        
        stop_routes = stop_times.merge(
            trips[['trip_id', 'route_id']],
            on='trip_id'
        ).groupby('stop_id')['route_id'].apply(set).to_dict()
        
        route_pairs = defaultdict(set)
        
        for stop_id, routes in stop_routes.items():
            routes_list = list(routes)
            for i, r1 in enumerate(routes_list):
                for r2 in routes_list[i+1:]:
                    route_pairs[(r1, r2)].add(stop_id)
        
        overlaps = []
        for (r1, r2), shared_stops in route_pairs.items():
            overlaps.append({
                'route1': r1,
                'route2': r2,
                'shared_stops': len(shared_stops),
                'shared_stop_ids': list(shared_stops)
            })
        
        return pd.DataFrame(overlaps).sort_values('shared_stops', ascending=False)
    
    def dead_zones(self, min_routes: int = 2) -> pd.DataFrame:
        """
        Identify areas with poor network connectivity.
        
        Args:
            min_routes: Minimum routes needed for good connectivity
            
        Returns:
            DataFrame of stops with poor connectivity
        """
        stop_times = self.loader.stop_times
        trips = self.loader.trips
        
        stop_routes = stop_times.merge(
            trips[['trip_id', 'route_id']],
            on='trip_id'
        ).groupby('stop_id')['route_id'].nunique().to_dict()
        
        stops = self.loader.stops.copy()
        stops['n_routes'] = stops['stop_id'].map(stop_routes).fillna(0)
        
        dead_zones = stops[stops['n_routes'] < min_routes].copy()
        
        return dead_zones.sort_values('n_routes')

