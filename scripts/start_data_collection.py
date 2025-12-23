#!/usr/bin/env python3
"""
Start continuous real-time data collection for COTA analysis.

Usage:
    python scripts/start_data_collection.py [--hours HOURS] [--interval MINUTES]
    
Examples:
    # Collect for 24 hours, snapshot every 5 minutes
    python scripts/start_data_collection.py --hours 24
    
    # Collect indefinitely, snapshot every 10 minutes
    python scripts/start_data_collection.py --interval 10
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loaders.data_collector import RealtimeDataCollector


def main():
    parser = argparse.ArgumentParser(description='Collect COTA real-time data')
    parser.add_argument('--hours', type=int, default=None,
                       help='Duration in hours (None = indefinitely)')
    parser.add_argument('--interval', type=int, default=2,
                       help='Snapshot interval in minutes (default: 2, recommended: 1-2 for bunching, 5 for general)')
    
    args = parser.parse_args()
    
    collector = RealtimeDataCollector(snapshot_interval_minutes=args.interval)
    
    if args.hours:
        print(f"Starting {args.hours}-hour data collection...")
        print(f"Snapshot interval: {args.interval} minutes")
        print("Press Ctrl+C to stop early\n")
        collector.run_continuous(duration_hours=args.hours)
    else:
        print("Starting continuous data collection (indefinite)...")
        print(f"Snapshot interval: {args.interval} minutes")
        print("Press Ctrl+C to stop\n")
        collector.run_continuous()


if __name__ == '__main__':
    main()

