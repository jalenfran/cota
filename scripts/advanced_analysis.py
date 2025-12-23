#!/usr/bin/env python3
"""
Advanced Quantitative Analysis Script
Performs sophisticated analysis using MIP, network analysis, and Monte Carlo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.loaders.gtfs import GTFSLoader
from src.loaders.census import CensusLoader
from src.analysis.metrics import RouteMetrics
from src.analysis.network_topology import TransitNetwork
from src.models.optimize import StopPlacementOptimizer
from src.models.mip_optimizer import MIPStopOptimizer
from src.models.monte_carlo import MonteCarloSimulator
from src.config.transit_params import get_cota_params


def main():
    print("=" * 80)
    print("Advanced Quantitative Analysis - COTA Transit System")
    print("=" * 80)
    print()

    # 0. Load Configuration
    print("0. Loading configuration...")
    loader = GTFSLoader()
    loader.load_all()
    params = get_cota_params(gtfs_loader=loader)
    print(f"   Operating days/year: {params['operating_days_per_year']}")
    print(f"   Ridership rate: {params['ridership_rate']*100:.1f}%")
    print(f"   Maintenance cost: {params['maintenance_cost_pct']*100:.0f}%")
    print()

    # 1. Load Data
    print("1. Loading data...")
    
    census = CensusLoader()
    population_grid = census.load_population_grid(include_transit_dependency=True)
    
    if 'zero_vehicle_households' not in population_grid.columns or population_grid['zero_vehicle_households'].sum() == 0:
        population_grid = census.load_population_grid(force_download=True, include_transit_dependency=True)
    
    if 'zero_vehicle_households' in population_grid.columns:
        total_transit_dep = population_grid['zero_vehicle_households'].sum()
        print(f"   Total transit-dependent households in grid: {total_transit_dep:,.0f}")
    else:
        print("   Warning: Transit dependency data not found, forcing re-download...")
        population_grid = census.load_population_grid(force_download=True, include_transit_dependency=True)
    
    # 2. Route Efficiency Analysis
    print("\n2. Route Efficiency Analysis...")
    metrics = RouteMetrics(loader)
    routes_summary = metrics.all_routes_summary()
    
    routes_summary['efficiency_score'] = (
        1 / routes_summary['directness'] * 0.4 +
        routes_summary['stops_per_km'] / routes_summary['stops_per_km'].max() * 0.3 +
        routes_summary['service_span_hours'] / routes_summary['service_span_hours'].max() * 0.3
    )
    routes_summary = routes_summary.sort_values('efficiency_score', ascending=False)
    
    print(f"   Top 5 Most Efficient Routes:")
    for _, route in routes_summary.head(5).iterrows():
        print(f"   - Route {route['route_name']}: score={route['efficiency_score']:.3f}, "
              f"directness={route['directness']:.2f}")
    
    inefficient = routes_summary[routes_summary['efficiency_score'] < routes_summary['efficiency_score'].median()]
    print(f"\n   Routes needing attention: {len(inefficient)}")
    
    # 3. Coverage Gap Analysis (Prioritizing Transit-Dependent)
    print("\n3. Coverage Gap Analysis (Prioritizing Transit-Dependent Populations)...")
    optimizer = StopPlacementOptimizer(
        existing_stops=loader.stops,
        population_grid=population_grid
    )
    
    gaps = optimizer.find_coverage_gaps(
        threshold_minutes=5,
        prioritize_transit_dependent=True,
        transit_dependent_weight=2.0
    )
    total_pop = population_grid['population'].sum()
    underserved = gaps['population'].sum()
    coverage_pct = (1 - underserved / total_pop) * 100
    
    print(f"   Coverage: {coverage_pct:.1f}%")
    print(f"   Underserved population: {underserved:,}")
    print(f"   Underserved areas: {len(gaps):,}")
    
    if 'zero_vehicle_households' in gaps.columns and 'zero_vehicle_households' in population_grid.columns:
        transit_dep_underserved = gaps['zero_vehicle_households'].sum()
        total_transit_dep = population_grid['zero_vehicle_households'].sum()
        if total_transit_dep > 0:
            print(f"   Underserved transit-dependent households: {transit_dep_underserved:,.0f}")
            print(f"   Transit dependency rate: {(transit_dep_underserved / total_transit_dep * 100):.1f}%")
    
    top_gaps = gaps.nlargest(50, 'priority_score')
    
    # 4. Network Topology
    print("\n4. Network Topology Analysis...")
    network = TransitNetwork(loader)
    
    transfers = network.transfer_opportunities(max_walk_meters=400)
    connectivity = network.route_connectivity()
    overlap = network.service_overlap()
    dead_zones = network.dead_zones(min_routes=2)
    
    print(f"   Transfer opportunities: {len(transfers):,}")
    print(f"   Route connections: {len(connectivity):,}")
    print(f"   Overlapping routes: {len(overlap):,}")
    print(f"   Dead zones: {len(dead_zones):,}")
    
    print(f"\n   Top 5 Transfer Opportunities:")
    for _, transfer in transfers.nlargest(5, 'n_transfers').iterrows():
        print(f"   - {transfer['stop1_id']} <-> {transfer['stop2_id']}: "
              f"{transfer['n_transfers']} routes, {transfer['distance_m']:.0f}m")
    
    # 5. Optimal Stop Placement (MIP)
    print("\n5. Optimal Stop Placement (Integer Programming)...")
    optimal_proposals = []
    
    try:
        candidate_locations = pd.DataFrame({
            'lat': top_gaps['lat'].values[:30],
            'lon': top_gaps['lon'].values[:30],
            'route_id': ['TBD'] * min(30, len(top_gaps))
        })
        
        mip_optimizer = MIPStopOptimizer(
            existing_stops=loader.stops,
            population_grid=population_grid,
            max_stops=10,
            transit_params=params
        )
        
        optimal_proposals = mip_optimizer.optimize(
            candidate_locations=candidate_locations,
            budget=100000,
            cost_per_stop=params['cost_per_stop']
        )
        
        print(f"   Optimal stops selected: {len(optimal_proposals)}")
        print(f"   Total population served: {sum(p.population_within_400m for p in optimal_proposals):,}")
        
        for i, prop in enumerate(optimal_proposals[:5], 1):
            print(f"\n   {i}. Location: ({prop.location[0]:.5f}, {prop.location[1]:.5f})")
            print(f"      Population: {prop.population_within_400m:,}")
            if hasattr(prop, '__dict__') and 'transit_dependent_households' in prop.__dict__:
                print(f"      Transit-dependent households: {prop.__dict__['transit_dependent_households']:,.0f}")
            print(f"      Est. boardings: {prop.estimated_daily_boardings:.0f}/day")
            print(f"      ROI: {prop.annual_roi:.1f}%")
    
    except (ImportError, RuntimeError, OSError) as e:
        print(f"   MIP solver unavailable: {e}")
        print("   Falling back to greedy algorithm...")
        
        if 'candidate_locations' not in locals():
            candidate_locations = pd.DataFrame({
                'lat': top_gaps['lat'].values[:30],
                'lon': top_gaps['lon'].values[:30],
                'route_id': ['TBD'] * min(30, len(top_gaps))
            })
        
        optimal_proposals = optimizer.propose_new_stops(
            candidate_locations=candidate_locations,
            n_stops=10,
            budget=100000,
            cost_per_stop=params['cost_per_stop']
        )
        print(f"   Greedy solution: {len(optimal_proposals)} stops")
    
    # 6. Monte Carlo Uncertainty
    print("\n6. Monte Carlo Uncertainty Quantification...")
    mc = MonteCarloSimulator(n_simulations=10000)
    
    if optimal_proposals:
        proposals_to_analyze = optimal_proposals[:5]
    else:
        print("   No proposals to analyze")
        proposals_to_analyze = []
    
    for i, prop in enumerate(proposals_to_analyze, 1):
        mc_result = mc.simulate_stop_proposal(
            population_nearby=prop.population_within_400m,
            cost_per_stop=prop.implementation_cost
        )
        
        print(f"\n   Stop {i} ROI Analysis (10,000 simulations):")
        print(f"      Mean ROI: {mc_result.mean:.1f}% ± {mc_result.std:.1f}%")
        print(f"      95% CI: [{mc_result.confidence_interval[0]:.1f}%, "
              f"{mc_result.confidence_interval[1]:.1f}%]")
        print(f"      5th percentile: {mc_result.percentiles[5]:.1f}%")
        print(f"      95th percentile: {mc_result.percentiles[95]:.1f}%")
    
    # 7. Statistical Analysis
    print("\n7. Statistical Analysis...")
    efficient_routes = routes_summary[routes_summary['efficiency_score'] > 
                                      routes_summary['efficiency_score'].median()]
    inefficient_routes = routes_summary[routes_summary['efficiency_score'] <= 
                                        routes_summary['efficiency_score'].median()]
    
    t_stat, p_value = stats.ttest_ind(
        efficient_routes['directness'],
        inefficient_routes['directness']
    )
    
    print(f"   Efficient vs Inefficient Routes:")
    print(f"      Efficient (n={len(efficient_routes)}): "
          f"directness={efficient_routes['directness'].mean():.3f}")
    print(f"      Inefficient (n={len(inefficient_routes)}): "
          f"directness={inefficient_routes['directness'].mean():.3f}")
    print(f"      T-test: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"      Significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("- Review results in notebooks/04_advanced_quant_analysis.ipynb")
    print("- Generate professional reports")
    print("- Collect real-time data for delay/bunching analysis")


if __name__ == '__main__':
    main()

