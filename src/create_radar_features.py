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
x, y, z = None, None, None
values, values_interpolated = None, None
daily_features = None
time_features = None

def create_radar_features(time_segments_file, grid, time_step, data_dir, output_dir, radar, parameters):
    
    # Global variables for ipython
    global time_segments, radar_sweeps, x, y, z, values, values_interpolated, time_features, daily_features

    # Read storm periods
    storm_periods = utils.parse_storm_periods(time_segments_file)

    # Loop over all storms
    for i, time in storm_periods.iterrows():
        start_reference_time = time["start_datetime"]
        end_reference_time = time["end_datetime"]

        print(f"\nProcessing storm day: {start_reference_time.date()}  {i+1}/{len(storm_periods)}")

        # Split time period into segments of length "time_step"
        time_segments = utils.create_time_segments(start_reference_time, end_reference_time, time_step)

        daily_features = {}
        # Loop over all time segments
        for reference_time in time_segments:
            print(reference_time.strftime('%H:%M'))
            time_features = {}

            # Compute Cartesian coordinates 
            filename = radar_utils.construct_radar_filename(reference_time, radar, parameters[0], data_dir, time_step)
            radar_sweeps = radar_utils.load_all_sweeps(filename)
            x, y, z = radar_utils.convert_all_sweeps_to_cartesian(radar_sweeps)

            for param in parameters:
                # Load data
                filename = radar_utils.construct_radar_filename(reference_time, radar, param, data_dir, time_step)
                radar_sweeps = radar_utils.load_all_sweeps(filename)
                dataset_key = radar_utils.PARAM_MAPPING[param]
                values = radar_utils.extract_parameter_values(radar_sweeps, dataset_key)

                # Interpolate to grid
                values_interpolated = radar_utils.interpolate_to_grid(x, y, z, values, grid)
                time_features[param] = values_interpolated

            # Store features for this time segment
            time_stamp = reference_time.strftime("%Hh%M")
            daily_features[time_stamp] = time_features

            
        date_label = start_reference_time.strftime('%Y-%m-%d')
        radar_utils.save_radar_features(daily_features, output_dir, date_label, grid)
        


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
    
