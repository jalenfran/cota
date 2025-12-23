"""
Integer Programming (MIP) optimizer for stop placement.
Replaces greedy algorithm with optimal solution using PuLP.
"""
import numpy as np
import pandas as pd
import shutil
from typing import List, Tuple, Optional

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

from src.models.optimize import StopProposal, haversine_distance_matrix
from src.config.transit_params import get_cota_params


def _get_cbc_solver():
    """Detect and return appropriate CBC solver command for PuLP."""
    system_cbc = shutil.which('cbc')
    if system_cbc:
        return pulp.COIN_CMD(path=system_cbc, msg=0)
    return pulp.PULP_CBC_CMD(msg=0)


class MIPStopOptimizer:
    """Optimal stop placement using Mixed Integer Programming"""
    
    def __init__(
        self,
        existing_stops: pd.DataFrame,
        population_grid: pd.DataFrame,
        max_walk_distance_km: float = 0.4,
        max_stops: int = 20,
        transit_params: Optional[dict] = None
    ):
        """
        Args:
            existing_stops: DataFrame with stop locations (stop_lat, stop_lon or lat, lon)
            population_grid: DataFrame with lat, lon, population columns
            max_walk_distance_km: Maximum walk distance (default 400m)
            max_stops: Maximum number of stops to place
            transit_params: Optional dict of transit parameters (from get_cota_params).
                          If None, uses default values.
        """
        if not PULP_AVAILABLE:
            raise ImportError("Install PuLP: pip install pulp")
        
        stops = existing_stops.copy()
        if 'lat' not in stops.columns:
            if 'stop_lat' in stops.columns:
                stops = stops.rename(columns={'stop_lat': 'lat', 'stop_lon': 'lon'})
            else:
                raise KeyError("Need lat/lon columns")
        
        self.existing_stops = stops[['lat', 'lon']].values
        
        cols_to_keep = ['lat', 'lon', 'population']
        if 'zero_vehicle_households' in population_grid.columns:
            cols_to_keep.append('zero_vehicle_households')
        self.population_grid = population_grid[cols_to_keep].copy()
        
        self.max_walk_distance = max_walk_distance_km
        self.max_stops = max_stops
        self.transit_params = transit_params
        
    def optimize(
        self,
        candidate_locations: pd.DataFrame,
        budget: float,
        cost_per_stop: float = 10000,
        min_population_per_stop: int = 100
    ) -> List[StopProposal]:
        """
        Solve optimal stop placement using MIP.
        
        Args:
            candidate_locations: DataFrame with lat, lon, route_id (optional)
            budget: Available budget
            cost_per_stop: Cost per stop installation
            min_population_per_stop: Minimum population to justify a stop
            
        Returns:
            List of StopProposal objects (optimal solution)
        """
        n_candidates = len(candidate_locations)
        n_population = len(self.population_grid)
        
        if n_candidates == 0:
            return []
        
        candidate_coords = candidate_locations[['lat', 'lon']].values
        pop_coords = self.population_grid[['lat', 'lon']].values
        
        distances = haversine_distance_matrix(
            pop_coords[:, 0], pop_coords[:, 1],
            candidate_coords[:, 0], candidate_coords[:, 1]
        )
        
        coverage = distances <= self.max_walk_distance
        
        population = self.population_grid['population'].values
        
        if 'zero_vehicle_households' in self.population_grid.columns:
            transit_dependent = self.population_grid['zero_vehicle_households'].values * 2.5
            weighted_population = population + transit_dependent * 2.0
        else:
            weighted_population = population
        
        prob = pulp.LpProblem("StopPlacement", pulp.LpMaximize)
        
        x = [pulp.LpVariable(f"stop_{i}", cat='Binary') for i in range(n_candidates)]
        y = [pulp.LpVariable(f"covered_{j}", cat='Binary') for j in range(n_population)]
        
        prob += pulp.lpSum([weighted_population[j] * y[j] for j in range(n_population)])
        
        prob += pulp.lpSum([cost_per_stop * x[i] for i in range(n_candidates)]) <= budget
        
        prob += pulp.lpSum([x[i] for i in range(n_candidates)]) <= self.max_stops
        
        for j in range(n_population):
            if weighted_population[j] < min_population_per_stop:
                continue
            
            prob += y[j] <= pulp.lpSum([x[i] for i in range(n_candidates) if coverage[j, i]])
        
        for i in range(n_candidates):
            pop_served = weighted_population[coverage[:, i]].sum()
            if pop_served < min_population_per_stop:
                prob += x[i] == 0
        
        try:
            solver = _get_cbc_solver()
            prob.solve(solver)
        except (OSError, FileNotFoundError) as e:
            raise RuntimeError(
                "CBC solver not available. Install CBC:\n"
                "  macOS: brew install cbc\n"
                "  Linux: sudo apt-get install coinor-cbc\n"
                "  Or ensure system CBC is in PATH"
            ) from e
        
        if prob.status != pulp.LpStatusOptimal:
            return []
        
        selected_indices = [i for i in range(n_candidates) if x[i].varValue == 1]
        
        proposals = []
        for idx in selected_indices:
            candidate = candidate_locations.iloc[idx]
            actual_pop_nearby = population[coverage[:, idx]].sum()
            transit_dep_nearby = 0
            if 'zero_vehicle_households' in self.population_grid.columns:
                transit_dep_nearby = (self.population_grid['zero_vehicle_households'].values * 
                                    coverage[:, idx]).sum()
            
            if self.transit_params is None:
                params = get_cota_params()
            else:
                params = self.transit_params
            ridership_rate = params['ridership_rate']
            est_boardings = actual_pop_nearby * ridership_rate
            annual_revenue = est_boardings * params['fare_per_trip'] * params['operating_days_per_year']
            annual_maintenance = cost_per_stop * params['maintenance_cost_pct']
            roi = ((annual_revenue - annual_maintenance) / cost_per_stop) * 100
            
            prop = StopProposal(
                location=(candidate['lat'], candidate['lon']),
                route_id=candidate.get('route_id', 'TBD'),
                estimated_daily_boardings=est_boardings,
                population_within_400m=int(actual_pop_nearby),
                coverage_gap_minutes=0,
                implementation_cost=cost_per_stop,
                annual_roi=roi,
                confidence_interval=(est_boardings * 0.7, est_boardings * 1.3)
            )
            
            if hasattr(prop, '__dict__'):
                prop.__dict__['transit_dependent_households'] = int(transit_dep_nearby)
            
            proposals.append(prop)
        
        return proposals

