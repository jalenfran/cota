"""
Census data loader for population and demographic data
Downloads ACS 5-year estimates and TIGER/Line shapefiles
"""
from pathlib import Path
from typing import Optional
import pandas as pd
import geopandas as gpd
import requests
import zipfile

try:
    import censusdata
    CENSUSDATA_AVAILABLE = True
except ImportError:
    CENSUSDATA_AVAILABLE = False


# Franklin County, Ohio FIPS codes
STATE_FIPS = '39'  # Ohio
COUNTY_FIPS = '049'  # Franklin County
ACS_YEAR = 2022  # Latest 5-year estimates


class CensusLoader:
    """Load census population data for Franklin County"""
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            current = Path(__file__).parent
            project_root = current.parent.parent
            data_dir = project_root / "data" / "census"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_tiger_shapefile(self, force: bool = False) -> Path:
        """Download TIGER/Line block group shapefile for Ohio"""
        extract_dir = self.data_dir / "tiger"
        shapefile_shp = extract_dir / f"tl_{ACS_YEAR}_{STATE_FIPS}_bg.shp"
        
        if shapefile_shp.exists() and not force:
            return shapefile_shp
        
        # Download from Census Bureau
        url = f"https://www2.census.gov/geo/tiger/TIGER{ACS_YEAR}/BG/tl_{ACS_YEAR}_{STATE_FIPS}_bg.zip"
        
        print(f"Downloading TIGER shapefile from {url}...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Extract zip
        zip_path = self.data_dir / f"tl_{ACS_YEAR}_{STATE_FIPS}_bg.zip"
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        zip_path.unlink()  # Remove zip file
        
        # Verify .shp file exists
        if not shapefile_shp.exists():
            shp_files = list(extract_dir.glob("*.shp"))
            if shp_files:
                shapefile_shp = shp_files[0]
            else:
                raise FileNotFoundError(f"Shapefile not found in {extract_dir}")
        
        return shapefile_shp
    
    def download_acs_population(self, force: bool = False) -> pd.DataFrame:
        """Download ACS 5-year population estimates by block group"""
        cache_file = self.data_dir / "acs_population.csv"
        
        if cache_file.exists() and not force:
            return pd.read_csv(cache_file)
        
        if not CENSUSDATA_AVAILABLE:
            raise ImportError(
                "Install censusdata: pip install censusdata\n"
                "Or use download_acs_population_api() for manual API calls"
            )
        
        data = censusdata.download(
            'acs5', ACS_YEAR,
            censusdata.censusgeo([
                ('state', STATE_FIPS),
                ('county', COUNTY_FIPS),
                ('block group', '*')
            ]),
            ['B01001_001E']
        )
        
        df = pd.DataFrame({
            'GEOID': [str(geo) for geo in data.index],
            'population': data['B01001_001E'].astype(int)
        })
        
        df.to_csv(cache_file, index=False)
        
        return df
    
    def download_acs_population_api(self, force: bool = False) -> pd.DataFrame:
        """Download ACS population using direct API"""
        cache_file = self.data_dir / "acs_population.csv"
        
        if cache_file.exists() and not force:
            return pd.read_csv(cache_file)
        
        base_url = "https://api.census.gov/data/2022/acs/acs5"
        
        params = {
            'get': 'B01001_001E,GEO_ID',  # Total population + GEOID
            'for': 'block group:*',
            'in': f'state:{STATE_FIPS} county:{COUNTY_FIPS}',
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data[1:], columns=data[0])
        df['B01001_001E'] = pd.to_numeric(df['B01001_001E'], errors='coerce').fillna(0).astype(int)
        df['GEOID'] = df['GEO_ID'].str.replace(r'^\d+US', '', regex=True)
        
        result = df[['GEOID', 'B01001_001E']].rename(columns={'B01001_001E': 'population'})
        
        result.to_csv(cache_file, index=False)
        
        return result
    
    def download_car_ownership_api(self, force: bool = False, use_tract_level: bool = True) -> pd.DataFrame:
        """
        Download ACS car ownership data (vehicles available per household).
        
        Uses tract-level data because block group data is suppressed for privacy.
        Tract-level data is then distributed proportionally to block groups.
        
        Variables:
        - B08201_001E: Total households
        - B08201_002E: Households with no vehicle available
        """
        cache_file = self.data_dir / "acs_car_ownership.csv"
        
        if cache_file.exists() and not force:
            return pd.read_csv(cache_file)
        
        base_url = "https://api.census.gov/data/2022/acs/acs5"
        
        # Try tract level first (less suppression)
        geo_level = 'tract:*' if use_tract_level else 'block group:*'
        geo_key = 'tract' if use_tract_level else 'block group'
        
        params = {
            'get': 'B08201_001E,B08201_002E,GEO_ID',
            'for': geo_level,
            'in': f'state:{STATE_FIPS} county:{COUNTY_FIPS}',
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if len(data) == 0:
            raise ValueError("Census API returned empty response")
        
        df = pd.DataFrame(data[1:], columns=data[0])
        df['B08201_001E'] = pd.to_numeric(df['B08201_001E'], errors='coerce').fillna(0).astype(int)
        df['B08201_002E'] = pd.to_numeric(df['B08201_002E'], errors='coerce').fillna(0).astype(int)
        df['GEOID'] = df['GEO_ID'].str.replace(r'^\d+US', '', regex=True)
        
        # Create tract-level GEOID for merging (ensure string types)
        df['state'] = df['state'].astype(str).str.zfill(2)
        df['county'] = df['county'].astype(str).str.zfill(3)
        df['tract'] = df['tract'].astype(str)
        
        if use_tract_level:
            df['tract_geoid'] = (df['state'] + df['county'] + df['tract']).astype(str)
        else:
            df['tract_geoid'] = (df['state'] + df['county'] + df['tract']).astype(str)
            df['bg_geoid'] = df['GEOID']
        
        df['total_households'] = df['B08201_001E']
        df['zero_vehicle_households'] = df['B08201_002E']
        df['transit_dependent_pct'] = (df['zero_vehicle_households'] / df['total_households'].replace(0, 1) * 100).fillna(0)
        
        total_zero_veh = df['zero_vehicle_households'].sum()
        total_hh = df['total_households'].sum()
        
        result = df[['GEOID', 'tract_geoid', 'total_households', 'zero_vehicle_households', 'transit_dependent_pct']].copy()
        
        result.to_csv(cache_file, index=False)
        
        return result
    
    def create_population_grid(
        self,
        method: str = 'centroids',
        grid_spacing_km: float = 0.5,
        include_transit_dependency: bool = True
    ) -> pd.DataFrame:
        """
        Create population grid from census block groups.
        
        Strategy: Use finest granularity available for each variable:
        - Population: Block groups (not suppressed, finest granularity)
        - Transit dependency: Tracts (block groups suppressed, distribute proportionally)
        
        Args:
            method: 'centroids' or 'grid'
            grid_spacing_km: Grid spacing for 'grid' method
            include_transit_dependency: Include car ownership data
        """
        shapefile_shp = self.download_tiger_shapefile()
        pop_data = self.download_acs_population_api()
        
        bg_shp = gpd.read_file(shapefile_shp)
        franklin_bg = bg_shp[bg_shp['COUNTYFP'] == COUNTY_FIPS].copy()
        
        franklin_bg['GEOID'] = franklin_bg['GEOID'].astype(str)
        pop_data['GEOID'] = pop_data['GEOID'].astype(str)
        pop_data['GEOID'] = pop_data['GEOID'].str.replace(r'^\d+US', '', regex=True)
        
        franklin_bg = franklin_bg.merge(
            pop_data[['GEOID', 'population']],
            on='GEOID',
            how='left'
        )
        franklin_bg['population'] = franklin_bg['population'].fillna(0).astype(int)
        
        if include_transit_dependency:
            car_data = self.download_car_ownership_api(use_tract_level=True)
            
            franklin_bg['tract_geoid'] = franklin_bg['GEOID'].astype(str).str[:11]
            
            tract_car_data = car_data.groupby('tract_geoid').agg({
                'zero_vehicle_households': 'sum',
                'total_households': 'sum'
            }).reset_index()
            tract_car_data['tract_geoid'] = tract_car_data['tract_geoid'].astype(str)
            tract_car_data['transit_dependent_pct'] = (
                tract_car_data['zero_vehicle_households'] / 
                tract_car_data['total_households'].replace(0, 1) * 100
            ).fillna(0)
            
            franklin_bg = franklin_bg.merge(
                tract_car_data[['tract_geoid', 'zero_vehicle_households', 'transit_dependent_pct']],
                on='tract_geoid',
                how='left'
            )
            
            tract_totals = franklin_bg.groupby('tract_geoid')['population'].sum()
            for tract_id in tract_totals.index:
                tract_mask = franklin_bg['tract_geoid'] == tract_id
                tract_pop = tract_totals[tract_id]
                if tract_pop > 0:
                    pop_share = franklin_bg.loc[tract_mask, 'population'] / tract_pop
                    zero_veh = franklin_bg.loc[tract_mask, 'zero_vehicle_households'].iloc[0] if tract_mask.any() else 0
                    franklin_bg.loc[tract_mask, 'zero_vehicle_households'] = (pop_share * zero_veh).fillna(0).astype(int)
            
            franklin_bg['zero_vehicle_households'] = franklin_bg['zero_vehicle_households'].fillna(0).astype(int)
            franklin_bg['transit_dependent_pct'] = franklin_bg['transit_dependent_pct'].fillna(0)
        
        if method == 'centroids':
            original_crs = franklin_bg.crs
            if franklin_bg.crs is None or franklin_bg.crs.is_geographic:
                franklin_bg_projected = franklin_bg.to_crs('EPSG:3857')
            else:
                franklin_bg_projected = franklin_bg.copy()
            
            franklin_bg_projected['centroid'] = franklin_bg_projected.geometry.centroid
            
            centroids_gdf = gpd.GeoDataFrame(
                geometry=franklin_bg_projected['centroid'],
                crs=franklin_bg_projected.crs
            ).to_crs('EPSG:4326')
            
            grid = pd.DataFrame({
                'lat': centroids_gdf.geometry.y,
                'lon': centroids_gdf.geometry.x,
                'population': franklin_bg['population'].values
            })
            
            if include_transit_dependency:
                grid['zero_vehicle_households'] = franklin_bg['zero_vehicle_households'].values
                grid['transit_dependent_pct'] = franklin_bg['transit_dependent_pct'].values
            
        elif method == 'grid':
            from shapely.geometry import Point
            import numpy as np
            
            bounds = franklin_bg.total_bounds
            lat_min, lon_min, lat_max, lon_max = bounds
            spacing_deg = grid_spacing_km * 0.009
            
            lats = np.arange(lat_min, lat_max, spacing_deg)
            lons = np.arange(lon_min, lon_max, spacing_deg)
            
            grid_points = []
            for lat in lats:
                for lon in lons:
                    point = Point(lon, lat)
                    intersecting = franklin_bg[franklin_bg.geometry.contains(point)]
                    if not intersecting.empty:
                        pop = intersecting.iloc[0]['population']
                        grid_points.append({
                            'lat': lat,
                            'lon': lon,
                            'population': pop
                        })
            
            grid = pd.DataFrame(grid_points)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        grid = grid[grid['population'] > 0].reset_index(drop=True)
        return grid
    
    def load_population_grid(self, force_download: bool = False, include_transit_dependency: bool = True) -> pd.DataFrame:
        """
        Load or create population grid with transit dependency data.
        
        Args:
            force_download: Force re-download of data
            include_transit_dependency: Include car ownership data
        """
        cache_file = self.data_dir / "population_grid.csv"
        
        if cache_file.exists() and not force_download:
            grid = pd.read_csv(cache_file)
            if include_transit_dependency and 'zero_vehicle_households' not in grid.columns:
                car_data = self.download_car_ownership_api()
                grid = self._merge_transit_dependency(grid, car_data)
                grid.to_csv(cache_file, index=False)
            elif include_transit_dependency and grid['zero_vehicle_households'].sum() == 0:
                car_data = self.download_car_ownership_api()
                grid = self._merge_transit_dependency(grid, car_data)
                grid.to_csv(cache_file, index=False)
            return grid
        
        grid = self.create_population_grid(method='centroids', include_transit_dependency=include_transit_dependency)
        grid.to_csv(cache_file, index=False)
        return grid
    
    def _merge_transit_dependency(self, grid: pd.DataFrame, car_data: pd.DataFrame) -> pd.DataFrame:
        """Merge transit dependency data into grid"""
        bg_shp = gpd.read_file(self.download_tiger_shapefile())
        franklin_bg = bg_shp[bg_shp['COUNTYFP'] == COUNTY_FIPS].copy()
        
        from src.models.optimize import haversine_distance_matrix
        
        if franklin_bg.crs is None or franklin_bg.crs.is_geographic:
            franklin_bg_projected = franklin_bg.to_crs('EPSG:3857')
        else:
            franklin_bg_projected = franklin_bg.copy()

        centroids = franklin_bg_projected.geometry.centroid
        centroids_gdf = gpd.GeoDataFrame(geometry=centroids, crs=franklin_bg_projected.crs).to_crs('EPSG:4326')

        grid_coords = grid[['lat', 'lon']].values
        bg_coords = pd.DataFrame({
            'lat': centroids_gdf.geometry.y,
            'lon': centroids_gdf.geometry.x
        }).values

        distances = haversine_distance_matrix(
            grid_coords[:, 0], grid_coords[:, 1],
            bg_coords[:, 0], bg_coords[:, 1]
        )
        nearest_bg_idx = distances.argmin(axis=1)
        
        franklin_bg['GEOID'] = franklin_bg['GEOID'].astype(str)
        franklin_bg['tract_geoid'] = franklin_bg['GEOID'].str[:11]
        
        if 'population' not in franklin_bg.columns:
            pop_data = self.download_acs_population()
            pop_data['GEOID'] = pop_data['GEOID'].astype(str)
            franklin_bg = franklin_bg.merge(pop_data, on='GEOID', how='left')
            franklin_bg['population'] = franklin_bg['population'].fillna(0).astype(int)
        
        car_data['tract_geoid'] = car_data['tract_geoid'].astype(str)
        
        tract_car_data = car_data.groupby('tract_geoid').agg({
            'zero_vehicle_households': 'sum',
            'total_households': 'sum'
        }).reset_index()
        tract_car_data['transit_dependent_pct'] = (
            tract_car_data['zero_vehicle_households'] / 
            tract_car_data['total_households'].replace(0, 1) * 100
        ).fillna(0)
        
        franklin_bg = franklin_bg.merge(
            tract_car_data[['tract_geoid', 'zero_vehicle_households', 'transit_dependent_pct']],
            on='tract_geoid',
            how='left'
        )
        
        tract_totals = franklin_bg.groupby('tract_geoid')['population'].sum()
        for tract_id in tract_totals.index:
            tract_mask = franklin_bg['tract_geoid'] == tract_id
            tract_pop = tract_totals[tract_id]
            if tract_pop > 0:
                pop_share = franklin_bg.loc[tract_mask, 'population'] / tract_pop
                zero_veh = franklin_bg.loc[tract_mask, 'zero_vehicle_households'].iloc[0] if tract_mask.any() else 0
                franklin_bg.loc[tract_mask, 'zero_vehicle_households'] = (pop_share * zero_veh).fillna(0).astype(int)
        
        franklin_bg['zero_vehicle_households'] = franklin_bg['zero_vehicle_households'].fillna(0).astype(int)
        franklin_bg['transit_dependent_pct'] = franklin_bg['transit_dependent_pct'].fillna(0)
        
        grid['zero_vehicle_households'] = franklin_bg.iloc[nearest_bg_idx]['zero_vehicle_households'].values
        grid['transit_dependent_pct'] = franklin_bg.iloc[nearest_bg_idx]['transit_dependent_pct'].values
        
        return grid.fillna(0)


if __name__ == '__main__':
    loader = CensusLoader()
    
    print("="*60)
    print("Census Data Loader Test")
    print("="*60)
    
    # Download and create grid
    grid = loader.load_population_grid()
    
    print(f"\nPopulation Grid Summary:")
    print(f"  Points: {len(grid):,}")
    print(f"  Total population: {grid['population'].sum():,}")
    print(f"  Average per point: {grid['population'].mean():.0f}")
    print(f"  Max per point: {grid['population'].max():,}")
    print(f"\nSample:")
    print(grid.head())

