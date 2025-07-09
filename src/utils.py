import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def parse_storm_periods(filename):
    storm_periods = pd.read_csv(filename, comment="#")
    storm_periods["start_datetime"] = pd.to_datetime(storm_periods["date"] + " " + storm_periods["start_time"])
    storm_periods["end_datetime"] = pd.to_datetime(storm_periods["date"] + " " + storm_periods["end_time"])
    # Convert to datetime
    storm_periods = storm_periods.drop(columns=["start_time", "date", "end_time"])
    return storm_periods

def create_time_segments(start_of_storm, end_of_storm, time_window):
    """
    Generate time segments at regular intervals between start and end times.

    Parameters
    ----------
    start_of_storm : datetime
        Start time
    end_of_storm : datetime  
        End time
    time_window : int
        Interval in minutes between segments
        
    Returns
    -------
    list of datetime
        Time segments from start to end at specified intervals
    """
    time_segments = []
    time = start_of_storm
    while time <= end_of_storm:
        time_segments.append(time)
        time += timedelta(minutes=time_window)
    return time_segments


def create_radar_grid(radar_lat, radar_lon, cell_size=8000, n_cells=10):
    """
    Create combined Cartesian/lat-lon grid for radar and lightning data

    Parameters:
    -----------
    radar_lat, radar_lon : float
        Radar location in degrees
    cell_size : float
        Grid cell size in meters (default: 8000m = 8km)
    n_cells : int
        Number of cells along each axis
    """
    grid_length = cell_size * n_cells  # meters

    # Create Cartesian grid (meters from radar)
    x_bounds_m = np.linspace(-grid_length/2, grid_length/2, n_cells+1)
    y_bounds_m = np.linspace(-grid_length/2, grid_length/2, n_cells+1)
    x_centers_m = (x_bounds_m[:-1] + x_bounds_m[1:]) / 2
    y_centers_m = (y_bounds_m[:-1] + y_bounds_m[1:]) / 2

    # Vertical levels (meters)
    z_levels_m = np.arange(200, 10000, 500)  # 200m to 10km, every 500m

    # Convert to lat/lon for lightning processing
    deg2m = 111000  # meters per degree
    grid_lats = radar_lat + y_bounds_m / deg2m
    grid_lons = radar_lon + x_bounds_m / (deg2m * np.cos(np.radians(radar_lat)))

    return {
        'radar_lat': radar_lat,
        'radar_lon': radar_lon,
        'cell_size_m': cell_size,
        'n_cells': n_cells,
        'grid_length_m': grid_length,
        'x_bounds_m': x_bounds_m,
        'y_bounds_m': y_bounds_m,
        'x_centers_m': x_centers_m,
        'y_centers_m': y_centers_m,
        'z_levels_m': z_levels_m,
        'grid_lats': grid_lats,
        'grid_lons': grid_lons,
    }