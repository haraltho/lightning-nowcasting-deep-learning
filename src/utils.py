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