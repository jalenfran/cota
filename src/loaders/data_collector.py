"""
Continuous real-time data collector for COTA GTFS-RT feeds.
Stores historical snapshots for delay analysis, bunching detection, and demand forecasting.
"""
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import time
import pandas as pd
import numpy as np

from src.loaders.realtime import GTFSRealtimeLoader
from src.analysis.realtime_delays import compute_realtime_delays_snapshot


class RealtimeDataCollector:
    """Collect and store real-time GTFS-RT data for historical analysis"""
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        snapshot_interval_minutes: int = 2
    ):
        """
        Args:
            output_dir: Directory to store collected data (default: data/realtime_history/)
            snapshot_interval_minutes: Minutes between snapshots (recommended: 1-2 for bunching, 5 for general analysis)
        """
        if output_dir is None:
            current = Path(__file__).parent
            project_root = current.parent.parent
            output_dir = project_root / "data" / "realtime_history"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = snapshot_interval_minutes * 60
        self.rt_loader = GTFSRealtimeLoader(use_live=True)
        
    def collect_snapshot(self) -> dict:
        """Collect single snapshot of current system state"""
        timestamp = datetime.now()
        
        try:
            vehicles = self.rt_loader.load_vehicle_positions()
            delays = compute_realtime_delays_snapshot(rt_loader=self.rt_loader)
            alerts = self.rt_loader.load_alerts()
            
            snapshot = {
                'timestamp': timestamp,
                'n_vehicles': len(vehicles),
                'n_delay_updates': len(delays),
                'n_alerts': len(alerts),
                'vehicles': vehicles,
                'delays': delays,
                'alerts': alerts
            }
            
            return snapshot
        except Exception as e:
            print(f"Error collecting snapshot at {timestamp}: {e}")
            return None
    
    def save_snapshot(self, snapshot: dict):
        """Save snapshot to parquet files"""
        if snapshot is None:
            return
        
        timestamp = snapshot['timestamp']
        date_str = timestamp.strftime('%Y%m%d')
        time_str = timestamp.strftime('%H%M%S')
        
        date_dir = self.output_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        prefix = f"snapshot_{date_str}_{time_str}"
        
        if not snapshot['vehicles'].empty:
            vehicles_path = date_dir / f"{prefix}_vehicles.parquet"
            snapshot['vehicles'].to_parquet(vehicles_path, index=False)
        
        if not snapshot['delays'].empty:
            delays_path = date_dir / f"{prefix}_delays.parquet"
            snapshot['delays'].to_parquet(delays_path, index=False)
        
        if snapshot['alerts']:
            alerts_path = date_dir / f"{prefix}_alerts.parquet"
            pd.DataFrame(snapshot['alerts']).to_parquet(alerts_path, index=False)
    
    def run_continuous(self, duration_hours: Optional[int] = None):
        """
        Run continuous data collection
        
        Args:
            duration_hours: How long to collect (None = indefinitely)
        """
        start_time = datetime.now()
        end_time = None
        if duration_hours:
            end_time = start_time + timedelta(hours=duration_hours)
        
        print(f"Starting data collection at {start_time}")
        print(f"Snapshot interval: {self.interval_seconds / 60:.0f} minutes")
        if end_time:
            print(f"Will collect until {end_time}")
        
        snapshot_count = 0
        
        while True:
            if end_time and datetime.now() >= end_time:
                print(f"Collection complete. Collected {snapshot_count} snapshots.")
                break
            
            snapshot = self.collect_snapshot()
            if snapshot:
                self.save_snapshot(snapshot)
                snapshot_count += 1
                print(f"[{datetime.now()}] Snapshot {snapshot_count}: "
                      f"{snapshot['n_vehicles']} vehicles, "
                      f"{snapshot['n_delay_updates']} delay updates")
            
            time.sleep(self.interval_seconds)
    
    def load_historical_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load collected historical data
        
        Args:
            start_date: Start date (YYYYMMDD format)
            end_date: End date (YYYYMMDD format)
            
        Returns:
            Combined DataFrame of all delays
        """
        all_delays = []
        
        date_dirs = sorted([d for d in self.output_dir.iterdir() if d.is_dir()])
        
        for date_dir in date_dirs:
            date_str = date_dir.name
            
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            
            delay_files = sorted(date_dir.glob("*_delays.parquet"))
            
            for delay_file in delay_files:
                try:
                    df = pd.read_parquet(delay_file)
                    all_delays.append(df)
                except Exception as e:
                    print(f"Error loading {delay_file}: {e}")
        
        if not all_delays:
            return pd.DataFrame()
        
        combined = pd.concat(all_delays, ignore_index=True)
        return combined


if __name__ == '__main__':
    collector = RealtimeDataCollector(snapshot_interval_minutes=2)
    
    print("Starting 24-hour data collection...")
    print("Snapshot interval: 2 minutes (recommended for bunching detection)")
    print("Press Ctrl+C to stop early")
    
    try:
        collector.run_continuous(duration_hours=24)
    except KeyboardInterrupt:
        print("\nCollection stopped by user")

