"""
GTFS Realtime Protobuf Loader
Parses VehiclePositions, TripUpdates, and Alerts feeds.
Supports both local files and live feeds from COTA servers.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import requests

try:
    from google.transit import gtfs_realtime_pb2
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False


# COTA GTFS-RT Feed URLs
COTA_FEEDS = {
    'vehicle_positions': 'https://gtfs-rt.cota.vontascloud.com/TMGTFSRealTimeWebService/Vehicle/VehiclePositions.pb',
    'trip_updates': 'https://gtfs-rt.cota.vontascloud.com/TMGTFSRealTimeWebService/TripUpdate/TripUpdates.pb',
    'alerts': 'https://gtfs-rt.cota.vontascloud.com/TMGTFSRealTimeWebService/Alert/Alerts.pb',
    'shapes': 'https://gtfs-rt.cota.vontascloud.com/TMGTFSRealTimeWebService/Shapes/Shapes.pb'
}


class GTFSRealtimeLoader:
    """Load GTFS-RT protobuf feeds from local files or live URLs"""
    
    def __init__(self, data_path: str = None, use_live: bool = False):
        if not PROTOBUF_AVAILABLE:
            raise ImportError("Install gtfs-realtime-bindings: pip install gtfs-realtime-bindings")
        
        if data_path is None:
            current = Path(__file__).parent
            project_root = current.parent.parent
            data_path = project_root / "data"
        
        self.path = Path(data_path)
        self.use_live = use_live
        
    def _fetch_feed(self, feed_type: str) -> bytes:
        """Fetch feed data from local file or live URL"""
        if self.use_live:
            url = COTA_FEEDS[feed_type]
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.content
            except requests.RequestException as e:
                raise ConnectionError(f"Failed to fetch live data: {e}")
        else:
            local_file = self.path / self._get_filename(feed_type)
            with open(local_file, 'rb') as f:
                return f.read()
    
    def _get_filename(self, feed_type: str) -> str:
        """Map feed type to local filename"""
        mapping = {
            'vehicle_positions': 'VehiclePositions.pb',
            'trip_updates': 'TripUpdates.pb',
            'alerts': 'Alerts.pb',
            'shapes': 'Shapes.pb'
        }
        return mapping[feed_type]
        
    def load_vehicle_positions(self) -> pd.DataFrame:
        """Parse VehiclePositions.pb into DataFrame"""
        feed = gtfs_realtime_pb2.FeedMessage()
        data = self._fetch_feed('vehicle_positions')
        feed.ParseFromString(data)
        
        records = []
        for entity in feed.entity:
            if entity.HasField('vehicle'):
                veh = entity.vehicle
                records.append({
                    'vehicle_id': veh.vehicle.id if veh.HasField('vehicle') else None,
                    'trip_id': veh.trip.trip_id if veh.HasField('trip') else None,
                    'route_id': veh.trip.route_id if veh.HasField('trip') else None,
                    'latitude': veh.position.latitude if veh.HasField('position') else None,
                    'longitude': veh.position.longitude if veh.HasField('position') else None,
                    'speed': veh.position.speed if veh.HasField('position') else None,
                    'timestamp': veh.timestamp if veh.HasField('timestamp') else None,
                })
        return pd.DataFrame(records)
    
    def load_trip_updates(self) -> pd.DataFrame:
        """Parse TripUpdates.pb into DataFrame"""
        feed = gtfs_realtime_pb2.FeedMessage()
        data = self._fetch_feed('trip_updates')
        feed.ParseFromString(data)
        
        records = []
        for entity in feed.entity:
            if entity.HasField('trip_update'):
                tu = entity.trip_update
                trip_id = tu.trip.trip_id if tu.HasField('trip') else None
                route_id = tu.trip.route_id if tu.HasField('trip') else None
                
                for update in tu.stop_time_update:
                    records.append({
                        'trip_id': trip_id,
                        'route_id': route_id,
                        'stop_id': update.stop_id if update.HasField('stop_id') else None,
                        'stop_sequence': update.stop_sequence if update.HasField('stop_sequence') else None,
                        'arrival_delay': update.arrival.delay if update.HasField('arrival') else None,
                        'departure_delay': update.departure.delay if update.HasField('departure') else None,
                    })
        return pd.DataFrame(records)
    
    def load_alerts(self) -> List[Dict[str, Any]]:
        """Parse Alerts.pb into list of dicts"""
        feed = gtfs_realtime_pb2.FeedMessage()
        data = self._fetch_feed('alerts')
        feed.ParseFromString(data)
        
        alerts = []
        for entity in feed.entity:
            if entity.HasField('alert'):
                alert = entity.alert
                alerts.append({
                    'header': alert.header_text.translation[0].text if alert.header_text.translation else None,
                    'description': alert.description_text.translation[0].text if alert.description_text.translation else None,
                    'cause': alert.cause,
                    'effect': alert.effect,
                })
        return alerts


    def save_feeds_locally(self, output_path: Optional[str] = None):
        """Download current feeds and save locally for offline analysis"""
        if not self.use_live:
            self.use_live = True
        
        save_path = Path(output_path) if output_path else self.path
        save_path.mkdir(parents=True, exist_ok=True)
        
        for feed_type, url in COTA_FEEDS.items():
            data = self._fetch_feed(feed_type)
            filename = self._get_filename(feed_type)
            with open(save_path / filename, 'wb') as f:
                f.write(data)


if __name__ == '__main__':
    print("="*60)
    print("COTA GTFS Realtime Data Loader")
    print("="*60)
    
    loader = GTFSRealtimeLoader(use_live=True)
    
    print("\nVehicle Positions:")
    vp = loader.load_vehicle_positions()
    print(f"  {len(vp)} vehicles currently tracked")
    if len(vp) > 0:
        print(vp.head())
    
    print("\nTrip Updates:")
    tu = loader.load_trip_updates()
    print(f"  {len(tu)} stop time updates")
    if len(tu) > 0:
        print(tu.head())
        print(f"\n  Average delay: {tu['arrival_delay'].mean():.1f} seconds")
    
    print("\nAlerts:")
    alerts = loader.load_alerts()
    print(f"  {len(alerts)} active alerts")
    for i, alert in enumerate(alerts[:3], 1):
        if alert.get('header'):
            print(f"  {i}. {alert['header']}")
    
    print("\n" + "="*60)

