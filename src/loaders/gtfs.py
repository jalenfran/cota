"""
GTFS Static Data Loader
Efficiently loads COTA schedule data with minimal validation.
"""
from pathlib import Path
from typing import Dict
import pandas as pd


class GTFSLoader:
    """Load GTFS static feed into pandas DataFrames"""
    
    def __init__(self, gtfs_path: str = None):
        if gtfs_path is None:
            current = Path(__file__).parent
            project_root = current.parent.parent
            gtfs_path = project_root / "data" / "cota.gtfs"
        self.path = Path(gtfs_path)
        self._data: Dict[str, pd.DataFrame] = {}
        
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all GTFS tables"""
        if not self.path.exists():
            raise FileNotFoundError(f"GTFS directory not found: {self.path}")
            
        tables = [
            'agency', 'routes', 'trips', 'stops', 
            'stop_times', 'calendar', 'calendar_dates',
            'shapes', 'transfers', 'fare_attributes', 'fare_rules'
        ]
        
        for table in tables:
            file = self.path / f"{table}.txt"
            if file.exists():
                self._data[table] = pd.read_csv(file)
                
        return self._data
    
    def __getitem__(self, key: str) -> pd.DataFrame:
        """Access loaded tables: loader['routes']"""
        if key not in self._data:
            file = self.path / f"{key}.txt"
            if file.exists():
                self._data[key] = pd.read_csv(file)
            else:
                raise KeyError(f"Table {key} not found")
        return self._data[key]
    
    @property
    def routes(self) -> pd.DataFrame:
        return self['routes']
    
    @property
    def stops(self) -> pd.DataFrame:
        return self['stops']
    
    @property
    def trips(self) -> pd.DataFrame:
        return self['trips']
    
    @property
    def stop_times(self) -> pd.DataFrame:
        return self['stop_times']
    
    @property
    def shapes(self) -> pd.DataFrame:
        return self['shapes']
    
    def summary(self) -> Dict[str, int]:
        """Get counts of key entities"""
        self.load_all()
        return {
            'routes': len(self._data['routes']),
            'stops': len(self._data['stops']),
            'trips': len(self._data['trips']),
            'stop_times': len(self._data['stop_times']),
            'shapes': len(self._data['shapes']),
        }


if __name__ == '__main__':
    loader = GTFSLoader()
    print("Loading GTFS data...")
    loader.load_all()
    print("\nSummary:")
    for key, count in loader.summary().items():
        print(f"  {key:15} {count:>8,}")

