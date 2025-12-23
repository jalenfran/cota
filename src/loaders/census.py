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
        
        print("Downloading ACS population data...")
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
        
        print("Downloading ACS population via API...")
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
    
    def create_population_grid(
        self,
        method: str = 'centroids',
        grid_spacing_km: float = 0.5
    ) -> pd.DataFrame:
        """Create population grid from census block groups"""
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
        
        if method == 'centroids':
            if franklin_bg.crs is None or franklin_bg.crs.is_geographic:
                franklin_bg = franklin_bg.to_crs('EPSG:3857')
            
            franklin_bg['centroid'] = franklin_bg.geometry.centroid
            franklin_bg_projected = franklin_bg.copy()
            
            centroids_gdf = gpd.GeoDataFrame(
                geometry=franklin_bg_projected['centroid'],
                crs=franklin_bg_projected.crs
            ).to_crs('EPSG:4326')
            
            grid = pd.DataFrame({
                'lat': centroids_gdf.geometry.y,
                'lon': centroids_gdf.geometry.x,
                'population': franklin_bg['population']
            })
            
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
    
    def load_population_grid(self, force_download: bool = False) -> pd.DataFrame:
        """Load or create population grid"""
        cache_file = self.data_dir / "population_grid.csv"
        
        if cache_file.exists() and not force_download:
            return pd.read_csv(cache_file)
        
        grid = self.create_population_grid(method='centroids')
        grid.to_csv(cache_file, index=False)
        return grid


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

