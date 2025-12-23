#!/usr/bin/env python3
"""
Quick analysis script - Run this NOW to get immediate insights.
Doesn't require historical real-time data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.gtfs import GTFSLoader
from src.loaders.census import CensusLoader
from src.analysis.metrics import RouteMetrics
from src.models.optimize import StopPlacementOptimizer
import pandas as pd


def main():
    print("=" * 70)
    print("COTA Quick Analysis - Static Data Only")
    print("=" * 70)
    print()
    
    # 1. Route Efficiency
    print("1. Loading GTFS data...")
    loader = GTFSLoader()
    loader.load_all()
    
    print("2. Calculating route efficiency metrics...")
    metrics = RouteMetrics(loader)
    routes_summary = metrics.all_routes_summary()
    
    print(f"\n   System Overview:")
    print(f"   - Routes: {len(routes_summary)}")
    print(f"   - Total daily trips: {routes_summary['trips_per_day'].sum():,}")
    print(f"   - Average directness: {routes_summary['directness'].mean():.2f}")
    
    inefficient = routes_summary[routes_summary['directness'] > routes_summary['directness'].quantile(0.75)]
    print(f"   - Routes needing attention: {len(inefficient)}")
    
    print(f"\n   Top 5 Least Efficient Routes:")
    top_inefficient = inefficient.nlargest(5, 'directness')
    for _, route in top_inefficient.iterrows():
        print(f"   - Route {route['route_name']}: directness={route['directness']:.2f}, "
              f"stops/km={route['stops_per_km']:.1f}")
    
    # 2. Coverage Gaps
    print("\n3. Loading census data...")
    census = CensusLoader()
    population_grid = census.load_population_grid()
    
    print("4. Analyzing coverage gaps...")
    optimizer = StopPlacementOptimizer(
        existing_stops=loader.stops,
        population_grid=population_grid
    )
    
    gaps = optimizer.find_coverage_gaps(threshold_minutes=10)
    
    print(f"\n   Coverage Analysis:")
    print(f"   - Underserved areas: {len(gaps):,}")
    print(f"   - Underserved population: {gaps['population'].sum():,}")
    print(f"   - Coverage percentage: {(1 - gaps['population'].sum() / population_grid['population'].sum()) * 100:.1f}%")
    
    # 3. Top Opportunities
    print("\n5. Identifying top opportunities...")
    top_gaps = gaps.nlargest(10, 'population')
    
    print(f"\n   Top 10 Underserved Areas:")
    for i, (_, gap) in enumerate(top_gaps.iterrows(), 1):
        print(f"   {i}. Population: {gap['population']:,}, "
              f"Walk time: {gap['walk_time_minutes']:.1f} min, "
              f"Location: ({gap['lat']:.4f}, {gap['lon']:.4f})")
    
    # 4. Quick Stop Proposals
    print("\n6. Generating stop proposals...")
    candidate_locations = pd.DataFrame({
        'lat': top_gaps['lat'].values,
        'lon': top_gaps['lon'].values,
        'route_id': ['TBD'] * len(top_gaps)
    })
    
    proposals = optimizer.propose_new_stops(
        candidate_locations=candidate_locations,
        n_stops=5,
        budget=50000,
        cost_per_stop=10000
    )
    
    print(f"\n   Top 5 Stop Proposals:")
    for i, prop in enumerate(proposals, 1):
        print(f"\n   {i}. Location: ({prop.location[0]:.5f}, {prop.location[1]:.5f})")
        print(f"      Population served: {prop.population_within_400m:,}")
        print(f"      Est. daily boardings: {prop.estimated_daily_boardings:.0f}")
        print(f"      Annual ROI: {prop.annual_roi:.1f}%")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("- Run notebooks/02_professional_recommendations.ipynb for full analysis")
    print("- Collect real-time data for 1-2 weeks for delay/bunching analysis")
    print("- Use Monte Carlo for uncertainty quantification")


if __name__ == '__main__':
    main()

