import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
import importlib
importlib.reload(utils)
from datetime import timedelta
import os
    
import lightning_utils

storm_periods = None
grid = None
lightning_data = None
time_segments = None
lightning_grids = None

def process_lightning(time_segments_file, grid_file, lead_time, time_step, time_window, output_dir):
    """
    Load lightning data for all time segments and save as daily target file.
    
    For each storm day, generate lightning targets by:
    1. Creating 10-minute time samples throughout the storm period.
    2. For each sample time T, count lightning strikes from T+lead_time to T+lead_time+time_width
    3. Aggregate lightning counts onto the predefined grid.
    4. Output all targets for each day as h5-file.

    Parameters:
    -----------
    time_segments_file: str
        path to file that holds the storm periods with columns: start_datetime, end_datetime
    grid_file: str
        path to file that contains the spatial grid definition
    lead_time: int
        number of minutes to predict into the future
    time_step: int
        time step of processing in minutes (10 minutes, since radar data refreshes every 10 minutes)
    time_window: int
        number of minutes to aggreate lightnings (e.g. 10 minutes)
    output_dir: str
        path that will hold the resulting h5-files.

    Returns:
    --------
    None
        Saves h5-files with lightning count grids for each storm day.
    
    """

    # Global variables for ipython
    global storm_periods, grid, lightning_data, time_segments, lightning_grids

    # Read storm periods
    storm_periods = utils.parse_storm_periods(time_segments_file)

    # Load spatial grid
    grid = np.load(grid_file)

    # Loop over all storms
    for _, time in storm_periods.iterrows():

        # Compute shifted time window for the target
        start_reference_time = time["start_datetime"]
        end_reference_time   = time["end_datetime"]
        start_nowcast_time   = start_reference_time + timedelta(minutes=lead_time)
        end_nowcast_time     = end_reference_time   + timedelta(minutes=lead_time+time_window)

        # Fetch lightning data per storm day from api, or read it from file if it already exists
        filename = output_dir + "api_data/" + start_reference_time.strftime("%Y-%m-%d") + ".csv"
        if os.path.exists(filename):
            print(f"Loading {filename}")
            lightning_data = pd.read_csv(filename)
        else:
            print("Fetching from API.")
            lightning_data = lightning_utils.call_lightning_api(start_nowcast_time, end_nowcast_time, grid)
            print(filename)
            lightning_data.to_csv(filename, index=False)
            print(f"Saved {filename} to disk.")

        # Split time period into segments of length "time_step_min"
        time_segments = utils.create_time_segments(start_reference_time, end_reference_time, time_step)
        
        # Split lightning data into time segments and map onto grid
        lightning_grids = lightning_utils.lightning_to_grid(lightning_data, grid, time_segments, lead_time, time_window)

        # Save data to disk
        date_label = start_reference_time.strftime('%Y-%m-%d')
        lightning_utils.save_lightning_targets(lightning_grids, output_dir, date_label, lead_time, time_window, grid)


if __name__ == "__main__":
    storms_filename = "../data/storm_periods.csv"
    grid_filename = "../data/radar_hurum_grid_10x10_8km_spacing.npz"
    output_dir = "../data/processed_data/lightning/"

    process_lightning(time_segments_file=storms_filename, 
                      grid_file=grid_filename,
                      lead_time=30,
                      time_step=10,
                      time_window=10,
                      output_dir=output_dir)