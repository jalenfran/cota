#!/usr/bin/env python3
"""
Quick test of census data integration
Run: python test_census.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.loaders.census import CensusLoader

print("="*60)
print("Testing Census Data Integration")
print("="*60)

try:
    loader = CensusLoader()
    
    print("\n1. Downloading/loading population grid...")
    grid = loader.load_population_grid(force_download=False)
    
    print(f"\nSuccess!")
    print(f"  Points: {len(grid):,}")
    print(f"  Total population: {grid['population'].sum():,}")
    print(f"  Average per point: {grid['population'].mean():.0f}")
    print(f"  Max per point: {grid['population'].max():,}")
    
    print(f"\nSample data:")
    print(grid.head(10))
    
    print(f"\nCensus integration ready.")
    print(f"  Data cached at: {loader.data_dir}")
    print(f"  Use in notebooks with: CensusLoader().load_population_grid()")
    
except Exception as e:
    print(f"\nError: {e}")
    print("\nMake sure you've installed:")
    print("  pip install geopandas shapely censusdata")
    sys.exit(1)

