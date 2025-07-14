import numpy as np
import importlib
import matplotlib.pyplot as plt

import utils
import radar_utils
from global_variables import RADARS
import local_variables as loc_vars
importlib.reload(utils)
importlib.reload(radar_utils)
importlib.reload(loc_vars)

time_segments = None
radar_sweeps = None

def create_radar_features(time_segments_file, grid, time_step, data_dir, output_dir, radar, parameters):
    
    # Global variables for ipython
    global time_segments, radar_sweeps

    # Read storm periods
    storm_periods = utils.parse_storm_periods(time_segments_file)

    # Loop over all storms
    for _, time in storm_periods.iterrows():
        start_reference_time = time["start_datetime"]
        end_reference_time = time["end_datetime"]

        # Split time period into segments of length "time_step"
        time_segments = utils.create_time_segments(start_reference_time, end_reference_time, time_step)

        daily_features = {}
        # Loop over all time segments
        for reference_time in time_segments:
            time_features = {}

            for param in parameters:
                
                # Load data
                filename = radar_utils.construct_radar_filename(reference_time, radar, param, data_dir, time_step)
                radar_sweeps = radar_utils.load_all_sweeps(filename)

                # Convert to Cartesian coordinates
                dataset_key = radar_utils.PARAM_MAPPING[param]
                x, y, z, values = radar_utils.convert_all_sweeps_to_cartesian(radar_sweeps, dataset_key)

                
                

                

                break
            break
        break
    







if __name__ == "__main__":
    storms_filename = "../data/storm_periods.csv"
    output_dir = "../data/processed_data/radar/"
    data_dir = loc_vars.RADAR_FILE_DIR
    parameters = ["dBZ", "ZDR", "KDP", "RhoHV"]

    # Compute the grid around the radar
    radar = RADARS["hurum"]
    grid = utils.create_radar_grid(radar["lat"], radar["lon"])

    create_radar_features(
        time_segments_file=storms_filename,
        grid=grid,
        time_step=10,
        data_dir=data_dir,
        output_dir=output_dir,
        radar=radar,
        parameters=parameters,
        )