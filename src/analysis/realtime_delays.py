"""
Realtime delay computation utilities.

GTFS static provides scheduled arrival_time per (trip_id, stop_sequence).
GTFS-RT TripUpdates provide actual timestamps (epoch seconds).
Computed delay = actual_sec_since_midnight - scheduled_sec_since_midnight
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import zoneinfo

import numpy as np
import pandas as pd

from google.transit import gtfs_realtime_pb2  # type: ignore

from src.loaders.gtfs import GTFSLoader
from src.loaders.realtime import GTFSRealtimeLoader


TZ = zoneinfo.ZoneInfo("America/New_York")


def _parse_gtfs_time(t: str) -> int:
    """Parse GTFS HH:MM:SS into seconds since midnight. Handles hours >= 24."""
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def _epoch_to_local_seconds(ts: int) -> Tuple[int, datetime]:
    """Convert epoch seconds to (seconds since local midnight, local datetime)."""
    dt = datetime.fromtimestamp(ts, tz=TZ)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((dt - midnight).total_seconds()), dt


def compute_realtime_delays_snapshot(
    gtfs_loader: Optional[GTFSLoader] = None,
    rt_loader: Optional[GTFSRealtimeLoader] = None,
) -> pd.DataFrame:
    """
    Compute realized delays for a single realtime snapshot.

    Returns:
        DataFrame with columns:
            - trip_id, route_id, stop_id, stop_sequence
            - actual_ts, actual_local_time, actual_sec_since_midnight
            - rt_arrival_delay, rt_departure_delay (raw GTFS-RT fields, often 0)
            - arrival_time (scheduled GTFS string)
            - sched_arrival_sec (scheduled seconds since midnight)
            - computed_delay_sec, computed_delay_min
    """
    # Static GTFS
    gtfs_loader = gtfs_loader or GTFSLoader()
    gtfs_loader.load_all()
    stop_times = gtfs_loader.stop_times.copy()

    stop_times = stop_times[["trip_id", "stop_id", "stop_sequence", "arrival_time"]].copy()
    stop_times["trip_id"] = stop_times["trip_id"].astype(str)
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)
    stop_times["sched_arrival_sec"] = stop_times["arrival_time"].apply(_parse_gtfs_time)

    # Realtime TripUpdates
    rt_loader = rt_loader or GTFSRealtimeLoader(use_live=True)
    feed = gtfs_realtime_pb2.FeedMessage()
    raw = rt_loader._fetch_feed("trip_updates")  # internal helper is OK for our codebase
    feed.ParseFromString(raw)

    records: list[Dict[str, Any]] = []

    for entity in feed.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        trip_id = str(tu.trip.trip_id)
        route_id = tu.trip.route_id

        for stu in tu.stop_time_update:
            # Prefer arrival.time, then departure.time
            actual_ts: Optional[int] = None
            if stu.HasField("arrival") and stu.arrival.HasField("time"):
                actual_ts = stu.arrival.time
            elif stu.HasField("departure") and stu.departure.HasField("time"):
                actual_ts = stu.departure.time

            if actual_ts is None:
                continue

            sec_since_midnight, dt_local = _epoch_to_local_seconds(actual_ts)

            records.append(
                {
                    "trip_id": trip_id,
                    "route_id": route_id,
                    "stop_id": stu.stop_id,
                    "stop_sequence": int(stu.stop_sequence) if stu.HasField("stop_sequence") else np.nan,
                    "actual_ts": actual_ts,
                    "actual_local_time": dt_local,
                    "actual_sec_since_midnight": sec_since_midnight,
                    "rt_arrival_delay": (
                        stu.arrival.delay
                        if stu.HasField("arrival") and stu.arrival.HasField("delay")
                        else np.nan
                    ),
                    "rt_departure_delay": (
                        stu.departure.delay
                        if stu.HasField("departure") and stu.departure.HasField("delay")
                        else np.nan
                    ),
                }
            )

    if not records:
        return pd.DataFrame()

    df_rt = pd.DataFrame(records)
    df_rt = df_rt[df_rt["stop_sequence"].notna()].copy()
    df_rt["stop_sequence"] = df_rt["stop_sequence"].astype(int)

    merged = pd.merge(
        df_rt,
        stop_times[["trip_id", "stop_sequence", "sched_arrival_sec", "arrival_time"]],
        on=["trip_id", "stop_sequence"],
        how="left",
        validate="m:1",
    )

    merged["computed_delay_sec"] = merged["actual_sec_since_midnight"] - merged["sched_arrival_sec"]
    merged["computed_delay_min"] = merged["computed_delay_sec"] / 60.0

    return merged


def summarize_delay_snapshot(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Produce high-level summary metrics for a delay snapshot.

    Returns:
        dict with keys like:
            - total_updates
            - delay_summary (pandas describe() as dict)
            - on_time_pct
            - routes_top_delay (DataFrame)
    """
    if df.empty or "computed_delay_sec" not in df.columns:
        return {
            "total_updates": 0,
            "delay_summary": {},
            "on_time_pct": np.nan,
            "routes_top_delay": pd.DataFrame(),
        }

    valid = df["computed_delay_sec"].dropna()
    summary = valid.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    on_time = (valid.abs() <= 300).mean() * 100.0  # within 5 minutes

    by_route = (
        df.dropna(subset=["computed_delay_sec"])
        .groupby("route_id")["computed_delay_sec"]
        .agg(count="count", mean="mean", median="median")
        .sort_values("mean", ascending=False)
    )

    return {
        "total_updates": int(len(valid)),
        "delay_summary": summary,
        "on_time_pct": float(on_time),
        "routes_top_delay": by_route,
    }


