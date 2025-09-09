from global_variables import RADARS
import utils
import local_variables as loc_vars
import numpy as np
import os
import sys
import shutil

# Configurations
run_dir = "../data/processed_data/run_6_leadtime15/"
os.makedirs(run_dir, exist_ok=True)
lightning_output_dir = run_dir + "lightning/"
radar_output_dir     = run_dir + "radar/"
radar_data_dir       = loc_vars.RADAR_FILE_DIR
storms_filename = "../data/config/storm_periods.csv"
shutil.copy(storms_filename, run_dir)
radar = RADARS["hurum"]
parameters = ["dBZ", "ZDR", "KDP", "RhoHV"]
lead_time   = 15 # minutes; time to predict in the future
time_step   = 10 # minutes; length of time segments (defined by radar sweeps)
time_window = 45 # minutes; length of time window for lightning aggregation

print("\n== Lightning Nowcasting Pipeline ==")
print(f"\nProcessing radar: {radar['label']}")

# Step 1: Create spatial grid around the radar
grid = utils.create_radar_grid(radar["lat"], radar["lon"])
np.savez(f"{run_dir}/grid.npz", **grid)

# Step 2: Prosess lightning
print("\nProcessing lightning targets...")
from create_lightning_targets import process_lightning
process_lightning(
    time_segments_file=storms_filename,
    grid=grid,
    lead_time=lead_time,
    time_step=time_step,
    time_window=time_window,
    output_dir=lightning_output_dir,
)
sys.exit("asdf")
# Step 3: Process radar data
print("\nProcessing radar features...")
from create_radar_features import create_radar_features
create_radar_features(
    time_segments_file=storms_filename,
    grid=grid,
    time_step=time_step,
    data_dir=radar_data_dir,
    output_dir=radar_output_dir,
    radar=radar,
    parameters=parameters,
    )

print("\nProcessing complete.")