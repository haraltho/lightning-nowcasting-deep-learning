from global_variables import RADARS
import utils
import local_variables as loc_vars

# Configurations
storms_filename = "../data/config/storm_periods.csv"
radar = RADARS["hurum"]
parameters = ["dBZ", "ZDR", "KDP", "RhoHV"]
root_dir = "../data/processed_data/"
lightning_output_dir = root_dir + "lightning/"
radar_output_dir     = root_dir + "radar/"
radar_data_dir       = loc_vars.RADAR_FILE_DIR
lead_time   = 30 # minutes; time to predict in the future
time_step   = 10 # minutes; length of time segments (defined by radar sweeps)
time_window = 10 # minutes; length of time window for lightning aggregation

print("\n== Lightning Nowcasting Pipeline ==")
print(f"\nProcessing radar: {radar['label']}")

# Step 1: Create spatial grid around the radar
grid = utils.create_radar_grid(radar["lat"], radar["lon"])

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