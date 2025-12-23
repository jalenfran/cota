"""
Transit system parameters and configuration
Calculates operating days from GTFS and provides default parameters
"""
from typing import Optional
from datetime import datetime
import pandas as pd


def calculate_operating_days(
    calendar: pd.DataFrame,
    calendar_dates: pd.DataFrame,
    year: int = 2025
) -> int:
    """
    Calculate annual operating days from GTFS calendar data.
    For partial-year feeds, extrapolates based on service pattern.
    
    Args:
        calendar: GTFS calendar.txt DataFrame
        calendar_dates: GTFS calendar_dates.txt DataFrame
        year: Year to calculate for
        
    Returns:
        Estimated number of operating days per year
    """
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    dates = pd.date_range(start, end, freq='D')
    
    operating_days = set()
    weekday_names = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    
    feed_start = None
    feed_end = None
    
    for _, row in calendar.iterrows():
        try:
            service_start = datetime.strptime(str(int(row['start_date'])), '%Y%m%d')
            service_end = datetime.strptime(str(int(row['end_date'])), '%Y%m%d')
            
            if feed_start is None or service_start < feed_start:
                feed_start = service_start
            if feed_end is None or service_end > feed_end:
                feed_end = service_end
        except (ValueError, KeyError):
            continue
        
        for date in dates:
            if service_start <= date <= service_end:
                weekday = date.weekday()
                if row.get(weekday_names[weekday], 0) == 1:
                    operating_days.add(date.date())
    
    for _, row in calendar_dates.iterrows():
        try:
            date = datetime.strptime(str(int(row['date'])), '%Y%m%d').date()
            exception_type = int(row['exception_type'])
            
            if exception_type == 1:
                operating_days.add(date)
            elif exception_type == 2:
                operating_days.discard(date)
        except (ValueError, KeyError):
            continue
    
    actual_days = len(operating_days)
    
    if feed_start and feed_end:
        feed_duration = (feed_end - feed_start).days + 1
        feed_duration_years = feed_duration / 365.25
        
        if feed_duration_years < 0.8:
            service_days_per_week = set()
            for _, row in calendar.iterrows():
                week_days = []
                if row.get('monday', 0) == 1:
                    week_days.append(0)
                if row.get('tuesday', 0) == 1:
                    week_days.append(1)
                if row.get('wednesday', 0) == 1:
                    week_days.append(2)
                if row.get('thursday', 0) == 1:
                    week_days.append(3)
                if row.get('friday', 0) == 1:
                    week_days.append(4)
                if row.get('saturday', 0) == 1:
                    week_days.append(5)
                if row.get('sunday', 0) == 1:
                    week_days.append(6)
                service_days_per_week.update(week_days)
            
            if len(service_days_per_week) > 0:
                days_per_week = len(service_days_per_week)
                estimated_annual = int(days_per_week * 52.14)
                return estimated_annual
    
    return actual_days


def get_cota_params(gtfs_loader=None, year: int = 2025) -> dict:
    """
    Get COTA transit parameters, calculating operating days from GTFS if available.
    
    Args:
        gtfs_loader: Optional GTFSLoader instance to calculate operating days
        year: Year for operating days calculation
        
    Returns:
        Dictionary of transit parameters
    """
    params = {
        'ridership_rate': 0.01,
        'operating_days_per_year': 260,
        'maintenance_cost_pct': 0.15,
        'fare_per_trip': 2.0,
        'cost_per_stop': 10000,
        'vehicle_cost_per_hour': 100.0
    }
    
    if gtfs_loader is not None:
        try:
            calendar = gtfs_loader['calendar']
            calendar_dates = gtfs_loader['calendar_dates']
            params['operating_days_per_year'] = calculate_operating_days(
                calendar, calendar_dates, year
            )
        except (KeyError, FileNotFoundError):
            pass
    
    return params


COTA_PARAMS = get_cota_params()

