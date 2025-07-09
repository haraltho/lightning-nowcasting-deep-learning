import numpy as np
import importlib

import utils
import radar_utils
from global_variables import RADARS
importlib.reload(utils)
importlib.reload(radar_utils)

time_segments = None

def create_radar_features(time_segments_file, grid, time_step, output_dir):
    
    # Global variables for ipython
    global time_segments

    # Read storm periods
    storm_periods = utils.parse_storm_periods(time_segments_file)

    # Loop over all storms
    for _, time in storm_periods.iterrows():
        start_reference_time = time["start_datetime"]
        end_reference_time = time["end_datetime"]

        # Split time period into segments of length "time_step"
        time_segments = utils.create_time_segments(start_reference_time, end_reference_time, time_step)
        break






if __name__ == "__main__":
    storms_filename = "../data/storm_periods.csv"
    output_dir = "../data/processed_data/radar/"

    # Compute the grid around the radar
    radar = RADARS["hurum"]
    grid = utils.create_radar_grid(radar["lat"], radar["lon"])

    create_radar_features(
        time_segments_file=storms_filename,
        grid=grid,
        time_step=10,
        output_dir=output_dir)