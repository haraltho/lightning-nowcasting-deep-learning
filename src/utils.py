import pandas as pd

def parse_storm_periods(filename):
    storm_periods = pd.read_csv(filename, comment="#")
    storm_periods["start_datetime"] = pd.to_datetime(storm_periods["date"] + " " + storm_periods["start_time"])
    storm_periods["end_datetime"] = pd.to_datetime(storm_periods["date"] + " " + storm_periods["end_time"])
    # Convert to datetime
    storm_periods = storm_periods.drop(columns=["start_time", "date", "end_time"])
    return storm_periods