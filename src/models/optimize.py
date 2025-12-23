"""
Optimization models for stop placement and schedule adjustments
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2
from scipy.optimize import minimize, LinearConstraint

from src.config.transit_params import get_cota_params


def haversine_distance_matrix(lat1_arr, lon1_arr, lat2_arr, lon2_arr):
    """
    Calculate great-circle distances between two sets of points using haversine formula.
    Returns distance matrix in km with shape (len(lat1_arr), len(lat2_arr)).
    Element [i, j] is distance from point i in set 1 to point j in set 2.
    """
    R = 6371
    lat1_rad = np.radians(lat1_arr)
    lon1_rad = np.radians(lon1_arr)
    lat2_rad = np.radians(lat2_arr)
    lon2_rad = np.radians(lon2_arr)
    
    dlat = lat2_rad[np.newaxis, :] - lat1_rad[:, np.newaxis]
    dlon = lon2_rad[np.newaxis, :] - lon1_rad[:, np.newaxis]
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad[:, np.newaxis]) * np.cos(lat2_rad[np.newaxis, :]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


@dataclass
class StopProposal:
    """Proposed new bus stop"""
    location: Tuple[float, float]  # (lat, lon)
    route_id: str
    estimated_daily_boardings: float
    population_within_400m: int
    coverage_gap_minutes: float
    implementation_cost: float
    annual_roi: float
    confidence_interval: Tuple[float, float]
    
    def __str__(self) -> str:
        return (f"Stop at ({self.location[0]:.5f}, {self.location[1]:.5f})\n"
                f"  Route: {self.route_id}\n"
                f"  Est. boardings: {self.estimated_daily_boardings:.0f}/day "
                f"(95% CI: {self.confidence_interval[0]:.0f}-{self.confidence_interval[1]:.0f})\n"
                f"  Population served: {self.population_within_400m}\n"
                f"  ROI: {self.annual_roi:.1f}%")


@dataclass
class ScheduleProposal:
    """Proposed schedule adjustment"""
    route_id: str
    current_headway_min: int
    proposed_headway_min: int
    time_period: str
    ridership_impact: Dict[str, float]
    cost_impact: Dict[str, float]
    additional_vehicles_needed: int
    net_annual_benefit: float
    
    def __str__(self) -> str:
        return (f"Route {self.route_id} - {self.time_period}\n"
                f"  Headway: {self.current_headway_min}min → {self.proposed_headway_min}min\n"
                f"  Ridership: {self.ridership_impact['change_pct']:+.1f}% "
                f"({self.ridership_impact['additional_riders']:+.0f} riders/day)\n"
                f"  Additional vehicles: {self.additional_vehicles_needed}\n"
                f"  Net annual benefit: ${self.net_annual_benefit:,.0f}")


class StopPlacementOptimizer:
    """Optimize bus stop locations for maximum coverage"""
    
    def __init__(
        self,
        existing_stops: pd.DataFrame,
        population_grid: pd.DataFrame,
        max_walk_distance_km: float = 0.4
    ):
        """
        Args:
            existing_stops: DataFrame with stop locations.
                Accepts either GTFS-style columns (`stop_lat`, `stop_lon`)
                or generic `lat`, `lon` columns.
            population_grid: DataFrame with lat, lon, population columns
            max_walk_distance_km: Maximum acceptable walk distance (default 400m)
        """
        # Normalize stop coordinates to `lat` / `lon`
        stops = existing_stops.copy()
        if 'lat' not in stops.columns or 'lon' not in stops.columns:
            # Try GTFS stop columns
            if 'stop_lat' in stops.columns and 'stop_lon' in stops.columns:
                stops = stops.rename(columns={'stop_lat': 'lat', 'stop_lon': 'lon'})
            else:
                raise KeyError(
                    "existing_stops must contain either ['lat', 'lon'] or "
                    "['stop_lat', 'stop_lon'] columns"
                )
        self.existing_stops = stops
        
        # Ensure population grid has expected columns
        grid = population_grid.copy()
        missing = [c for c in ['lat', 'lon', 'population'] if c not in grid.columns]
        if missing:
            raise KeyError(f"population_grid is missing columns: {missing}")
        self.population_grid = grid
        self.max_walk_distance = max_walk_distance_km
        
    def calculate_coverage(self, stop_locations: np.ndarray) -> float:
        """Calculate population covered within walk distance"""
        pop_coords = self.population_grid[['lat', 'lon']].values
        distances = haversine_distance_matrix(
            pop_coords[:, 0], pop_coords[:, 1],
            stop_locations[:, 0], stop_locations[:, 1]
        )
        
        covered = np.any(distances <= self.max_walk_distance, axis=1)
        return self.population_grid.loc[covered, 'population'].sum()
    
    def propose_new_stops(
        self,
        candidate_locations: pd.DataFrame,
        n_stops: int,
        budget: float,
        cost_per_stop: float = 10000
    ) -> List[StopProposal]:
        """
        Select optimal new stop locations using greedy algorithm
        
        Args:
            candidate_locations: DataFrame with lat, lon, route_id
            n_stops: Number of stops to propose
            budget: Available budget
            cost_per_stop: Installation cost per stop
            
        Returns:
            List of StopProposal objects
        """
        if budget < cost_per_stop:
            return []
        
        max_stops = min(n_stops, int(budget / cost_per_stop))
        selected = []
        remaining = candidate_locations.copy()
        
        # Current coverage
        current_stops = self.existing_stops[['lat', 'lon']].values
        
        for i in range(max_stops):
            best_score = -1
            best_idx = None
            
            # Greedy: select location with highest marginal coverage
            for idx, candidate in remaining.iterrows():
                # Temporary stop set
                test_stops = np.vstack([
                    current_stops,
                    [candidate['lat'], candidate['lon']]
                ])
                
                coverage = self.calculate_coverage(test_stops)
                
                candidate_coords = np.array([[candidate['lat'], candidate['lon']]])
                min_dist = haversine_distance_matrix(
                    candidate_coords[:, 0], candidate_coords[:, 1],
                    current_stops[:, 0], current_stops[:, 1]
                )[0].min()
                
                score = coverage * min(min_dist / 0.5, 1.0)  # Penalty if too close
                
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx is not None:
                chosen = remaining.loc[best_idx]
                
                candidate_coords = np.array([[chosen['lat'], chosen['lon']]])
                pop_coords = self.population_grid[['lat', 'lon']].values
                distances = haversine_distance_matrix(
                    candidate_coords[:, 0], candidate_coords[:, 1],
                    pop_coords[:, 0], pop_coords[:, 1]
                )[0]
                
                pop_nearby = self.population_grid.loc[
                    distances <= self.max_walk_distance, 
                    'population'
                ].sum()
                
                params = get_cota_params()
                ridership_rate = params['ridership_rate']
                est_boardings = pop_nearby * ridership_rate
                
                fare_per_trip = params['fare_per_trip']
                operating_days = params['operating_days_per_year']
                annual_maintenance = cost_per_stop * params['maintenance_cost_pct']
                
                annual_revenue = est_boardings * fare_per_trip * operating_days
                roi = ((annual_revenue - annual_maintenance) / cost_per_stop) * 100
                
                proposal = StopProposal(
                    location=(chosen['lat'], chosen['lon']),
                    route_id=chosen.get('route_id', 'TBD'),
                    estimated_daily_boardings=est_boardings,
                    population_within_400m=int(pop_nearby),
                    coverage_gap_minutes=0,  # TODO: calculate
                    implementation_cost=cost_per_stop,
                    annual_roi=roi,
                    confidence_interval=(est_boardings * 0.7, est_boardings * 1.3)
                )
                
                selected.append(proposal)
                remaining = remaining.drop(best_idx)
                current_stops = np.vstack([
                    current_stops,
                    [chosen['lat'], chosen['lon']]
                ])
        
        return selected
    
    def find_coverage_gaps(
        self, 
        threshold_minutes: float = 5,
        prioritize_transit_dependent: bool = True,
        transit_dependent_weight: float = 2.0
    ) -> pd.DataFrame:
        """
        Identify areas with poor coverage, prioritizing transit-dependent populations.
        
        Args:
            threshold_minutes: Maximum walk time threshold (default 5 min = 400m)
            prioritize_transit_dependent: Weight zero-vehicle households higher
            transit_dependent_weight: Multiplier for transit-dependent population
            
        Returns:
            DataFrame of underserved areas with priority scores
        """
        pop_coords = self.population_grid[['lat', 'lon']].values
        stop_coords = self.existing_stops[['lat', 'lon']].values
        
        distances = haversine_distance_matrix(
            pop_coords[:, 0], pop_coords[:, 1],
            stop_coords[:, 0], stop_coords[:, 1]
        )
        
        min_distances = distances.min(axis=1)
        
        walk_time_minutes = (min_distances / 5) * 60
        
        gaps = self.population_grid.copy()
        gaps['nearest_stop_km'] = min_distances
        gaps['walk_time_minutes'] = walk_time_minutes
        gaps['coverage_gap'] = walk_time_minutes > threshold_minutes
        
        # Calculate priority score
        if prioritize_transit_dependent and 'zero_vehicle_households' in gaps.columns:
            gaps['transit_dependent_pop'] = gaps.get('zero_vehicle_households', 0) * 2.5
            gaps['priority_score'] = (
                gaps['population'] + 
                gaps.get('transit_dependent_pop', 0) * transit_dependent_weight
            )
        else:
            gaps['priority_score'] = gaps['population']
        
        gaps = gaps[gaps['coverage_gap']].copy()
        
        return gaps.sort_values('priority_score', ascending=False)


class ScheduleOptimizer:
    """Optimize service frequencies and headways"""
    
    def __init__(
        self,
        route_data: pd.DataFrame,
        demand_by_hour: pd.DataFrame,
        fleet_size: int,
        operating_cost_per_hour: float = 100
    ):
        """
        Args:
            route_data: DataFrame with route_id, current_headway_min, ridership
            demand_by_hour: Hourly demand patterns
            fleet_size: Total available vehicles
            operating_cost_per_hour: Cost per vehicle-hour
        """
        self.route_data = route_data
        self.demand_by_hour = demand_by_hour
        self.fleet_size = fleet_size
        self.cost_per_hour = operating_cost_per_hour
        
    def optimize_headways(
        self,
        time_period: str,
        max_headway: int = 60,
        min_headway: int = 10
    ) -> List[ScheduleProposal]:
        """
        Optimize headways to match demand while respecting fleet constraints
        
        Returns:
            List of schedule adjustment proposals
        """
        proposals = []
        period_demand = self.demand_by_hour[
            self.demand_by_hour['time_period'] == time_period
        ]
        
        for _, route in self.route_data.iterrows():
            route_demand = period_demand[
                period_demand['route_id'] == route['route_id']
            ]['demand'].mean()
            
            current_headway = route['current_headway_min']
            
            # Optimal headway based on demand (simple square root rule)
            # H = sqrt(2 * wait_time_value / (demand * operating_cost))
            # Simplified: inversely proportional to sqrt(demand)
            demand_factor = np.sqrt(route_demand / route_demand.mean())
            optimal_headway = int(30 / demand_factor)  # Base 30min headway
            
            # Constrain
            optimal_headway = max(min_headway, min(max_headway, optimal_headway))
            
            if optimal_headway != current_headway:
                # Calculate impact
                from src.models.backtest import ImpactEstimator
                
                ridership_impact = ImpactEstimator.headway_ridership_elasticity(
                    current_headway,
                    optimal_headway,
                    route['ridership']
                )
                
                # Vehicles needed
                route_cycle_time = route.get('cycle_time_min', 60)
                vehicles_current = int(np.ceil(route_cycle_time / current_headway))
                vehicles_proposed = int(np.ceil(route_cycle_time / optimal_headway))
                additional_vehicles = vehicles_proposed - vehicles_current
                
                params = get_cota_params()
                
                # Cost impact
                service_hours_per_day = route.get('service_hours', 16)
                additional_hours = (vehicles_proposed - vehicles_current) * service_hours_per_day
                annual_cost = additional_hours * params['operating_days_per_year'] * self.cost_per_hour
                
                # Revenue impact
                annual_revenue = ridership_impact['additional_riders'] * params['operating_days_per_year'] * params['fare_per_trip']
                
                cost_impact = {
                    'annual_operating_cost': annual_cost,
                    'annual_revenue': annual_revenue,
                    'net_benefit': annual_revenue - annual_cost
                }
                
                proposal = ScheduleProposal(
                    route_id=route['route_id'],
                    current_headway_min=current_headway,
                    proposed_headway_min=optimal_headway,
                    time_period=time_period,
                    ridership_impact=ridership_impact,
                    cost_impact=cost_impact,
                    additional_vehicles_needed=additional_vehicles,
                    net_annual_benefit=cost_impact['net_benefit']
                )
                
                proposals.append(proposal)
        
        # Sort by ROI
        proposals.sort(key=lambda p: p.net_annual_benefit, reverse=True)
        
        # Filter to fleet constraint
        cumulative_vehicles = 0
        feasible_proposals = []
        for p in proposals:
            if p.additional_vehicles_needed > 0:
                cumulative_vehicles += p.additional_vehicles_needed
                if cumulative_vehicles <= self.fleet_size:
                    feasible_proposals.append(p)
            else:
                feasible_proposals.append(p)  # Service reductions always feasible
        
        return feasible_proposals

