#!/usr/bin/env python3
"""
Simplified data collector that can run on any platform.
Works on: Local machine, Railway, Render, Heroku, etc.
"""
import sys
import time
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.data_collector import RealtimeDataCollector


def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    print('\nShutting down gracefully...')
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == '__main__':
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    collector = RealtimeDataCollector(snapshot_interval_minutes=interval)
    
    print("=" * 60)
    print("COTA Real-Time Data Collector")
    print("=" * 60)
    print(f"Snapshot interval: {interval} minutes")
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
        sys.exit(1)

