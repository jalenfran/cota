#!/usr/bin/env python3
"""
Simplified data collector for containerized deployment.
Collects COTA real-time data continuously.
"""
import os
import sys
import time
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, '/app/src')

from src.loaders.data_collector import RealtimeDataCollector


def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    print('\nShutting down gracefully...')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == '__main__':
    # Get configuration from environment variables
    snapshot_interval = int(os.getenv('SNAPSHOT_INTERVAL_MINUTES', '2'))
    alert_interval = int(os.getenv('ALERT_INTERVAL_MINUTES', '15'))
    data_dir = os.getenv('DATA_DIR', '/data/realtime_history')
    
    collector = RealtimeDataCollector(
        output_dir=data_dir,
        snapshot_interval_minutes=snapshot_interval,
        alert_interval_minutes=alert_interval
    )
    
    print("=" * 60)
    print("COTA Real-Time Data Collector (Containerized)")
    print("=" * 60)
    print(f"Snapshot interval: {snapshot_interval} minutes")
    print(f"Alert collection: {alert_interval} minutes")
    print(f"Data directory: {collector.output_dir}")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    try:
        collector.run_continuous()
    except KeyboardInterrupt:
        print("\nCollection stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

