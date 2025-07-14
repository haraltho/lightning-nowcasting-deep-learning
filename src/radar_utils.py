from datetime import timedelta
import glob
import xarray as xr


def construct_radar_filename(reference_time, radar, parameter, data_dir, time_step):
    """
    Find radar file for given time and parameter with fallback to earlier times.
   
    Parameters
    ----------
    reference_time : datetime
        Target time for radar data
    radar : dict
        Radar configuration with 'label' key
    parameter : str
        Radar parameter ('dBZ', 'ZDR', 'KDP', 'RhoHV')
    data_dir : str
        Base radar data directory
    time_step : int
        Time step in minutes for fallback attempts
       
    Returns
    -------
    str
        Full path to radar file
       
    Notes
    -----
    Tries reference_time-time_step, then -2*time_step, then -3*time_step
    if files are missing.
    """

    # A radar scan that finishes at 08:00 is stored in the file with the timestamp 07:50.
    # If that file doesn't exist, try time step before.

    for time_offset in [-time_step, -2*time_step, -3*time_step]:
        reference_time = reference_time + timedelta(minutes=time_offset)

        date_path = reference_time.strftime("%Y/%m/%d")
        time_base = reference_time.strftime("%Y%m%d%H%M")
        
        pattern = f"{data_dir}{date_path}/rainbow5/{radar['label']}/240km_12ele_DP_PROD011.vol/{time_base}????{parameter}.vol"
        matching_files = glob.glob(pattern)

        if not matching_files:
            print(f"Warning: No file found for {reference_time.strftime('%Y-%m-%d %H:%M')}. Trying earlier time step.")
        else:
            return matching_files[0]
        
    raise FileNotFoundError(f"No radar file found for reference time or up to 30min earlier.")


def load_all_sweeps(filename):
    """Load all 12 sweeps for one parameter"""
    sweeps = []
    for i in range(12):
        ds = xr.open_dataset(filename, group=f"sweep_{i}", engine="rainbow")
        sweeps.append(ds)
    return sweeps


