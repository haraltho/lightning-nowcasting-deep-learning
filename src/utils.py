import pandas as pd
import global_variables as global_vars
import local_variables as local_vars
import requests
import numpy as np
from datetime import datetime, timedelta
import io

def parse_storm_periods(filename):
    storm_periods = pd.read_csv(filename, comment="#")
    storm_periods["start_datetime"] = pd.to_datetime(storm_periods["date"] + " " + storm_periods["start_time"])
    storm_periods["end_datetime"] = pd.to_datetime(storm_periods["date"] + " " + storm_periods["end_time"])
    # Convert to datetime
    storm_periods = storm_periods.drop(columns=["start_time", "date", "end_time"])
    return storm_periods

def call_lightning_api(start_time, end_time, grid):
    """
    Fetch lightning data from FROST API for a given time period and spatial grid.
    
    Creates a bounding box from the grid coordinates and retrieves all lightning 
    strikes within that area and time window from the Norwegian Meteorological 
    Institute's FROST lightning API.
    
    Parameters
    ----------
    start_time : datetime
        Start of the time period to fetch lightning data
    end_time : datetime  
        End of the time period to fetch lightning data
    grid : numpy npz file
        Loaded grid file containing 'grid_lats' and 'grid_lons' arrays
        defining the spatial boundaries
        
    Returns
    -------
    pandas.DataFrame
        Lightning data with columns including lat, lon, timestamp components,
        peak_current, cloud type (0=cloud-to-ground, 1=intracloud), and
        various detection metadata
        
    Notes
    -----
    Requires FROST_CLIENT_ID in local_vars and FROST_ENDPOINT in global_vars.
    """

    # Create box around radar defined by the grid
    min_lat = np.min(grid["grid_lats"])
    max_lat = np.max(grid["grid_lats"])
    min_lon = np.min(grid["grid_lons"])
    max_lon = np.max(grid["grid_lons"])
    polygon = f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"

    # API parameters
    client_id = local_vars.FROST_CLIENT_ID
    endpoint = global_vars.FROST_ENDPOINT

    # Convert datetime to API string
    time_range = f"{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    
    columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 'nanoseconds',
            'lat', 'lon', 'peak_current', 'multi', 'nsens', 'dof', 'angle', 
            'major', 'minor', 'chi2', 'rt', 'ptz', 'mrr', 'cloud', 'aI', 'sI', 'tI']

    params = {
            'referencetime': time_range,
            'geometry': polygon
        }
    
    data = requests.get(endpoint, params=params, auth=(client_id, ''))

    lightning_data = pd.read_csv(io.StringIO(data.text), sep=' ', names=columns)
    lightning_data.reset_index()
    return lightning_data

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



def lightning_to_grid(lightning_data, grid, time_segments, lead_time, time_step):
    pass
    


