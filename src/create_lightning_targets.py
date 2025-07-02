import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils
    
storm_periods = None
grid = None

def process_lightning(time_segments_file, grid_file):

    # Global variables for ipython
    global storm_periods, grid

    # Read storm periods
    storm_periods = utils.parse_storm_periods(time_segments_file)

    # Load grid
    grid = np.load(grid_file)

if __name__ == "__main__":
    time_filename = "../data/storm_periods.csv"
    grid_filename = "../data/radar_hurum_grid_10x10_8km_spacing.npz"
    process_lightning(time_segments_file=time_filename, 
                      grid_file=grid_filename)