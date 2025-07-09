import numpy as np
import sys
import os
from global_variables import RADARS
from utils import create_radar_grid

def main():
    radar = RADARS["hurum"]
    grid = create_radar_grid(radar["lat"], radar["lon"])
    
    output_file = '../data/radar_hurum_grid_10x10_8km_spacing.npz'

    np.savez(output_file, **grid)
    print(f"Grid saved to {output_file}")
    print(f"Grid extent: ±{grid['grid_length_m']/2000:.0f} km")

if __name__ == "__main__":
    main()