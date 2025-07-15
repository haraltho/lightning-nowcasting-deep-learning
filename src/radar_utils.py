from datetime import timedelta
import glob
import xarray as xr
import numpy as np

PARAM_MAPPING = {
    "dBZ": "DBZH",
    "ZDR": "ZDR",
    "KDP": "KDP",
    "RhoHV": "RHOHV"
}


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
    """Load all 12 sweeps from the file."""

    sweeps = []
    for i in range(12):
        ds = xr.open_dataset(filename, group=f"sweep_{i}", engine="rainbow")
        sweeps.append(ds)
    return sweeps


def spherical_to_cartesian_3D(sweep):
    """
    Convert radar field from spherical to 3D Cartesian coordinates.
    
    Parameters
    ----------
    sweep : xarray.Dataset
        Single radar sweep containing range, azimuth, and elevation data
        
    Returns
    -------
    tuple of numpy.ndarray
        (x, y, z) coordinates in meters from radar location
        Each array has shape (n_azimuth, n_range)
        - x: East-West distance (positive = East)
        - y: North-South distance (positive = North) 
        - z: Height above radar (positive = up)
        
    Notes
    -----
    Uses standard meteorological coordinate system where:
    - Azimuth 0° = North, 90° = East
    - Elevation angle measured from horizontal
    """

    range_vals = sweep.range.values
    azimuth_vals = sweep.azimuth.values
    elevation_angle = sweep.sweep_fixed_angle.values

    R_mesh, Az_mesh = np.meshgrid(range_vals, azimuth_vals)

    phi = np.pi/2 - np.radians(elevation_angle) # Convert to zenith angle
    theta = np.radians(Az_mesh)

    x = R_mesh * np.sin(phi) * np.sin(theta)
    y = R_mesh * np.sin(phi) * np.cos(theta)
    z = R_mesh * np.cos(phi)

    return x, y, z


def convert_all_sweeps_to_cartesian(radar_sweeps):
    """
    Convert all radar sweeps to Cartesian coordinates and flatten for interpolation.
   
    Parameters
    ----------
    radar_sweeps : list of xarray.Dataset
        List of 12 elevation sweeps
       
    Returns
    -------
    tuple of numpy.ndarray
        (x, y, z) - flattened arrays ready for griddata interpolation
    """
    
    all_x, all_y, all_z = [], [], []

    for sweep in radar_sweeps:
        x, y, z = spherical_to_cartesian_3D(sweep)

        all_x.extend(x.flatten())
        all_y.extend(y.flatten())
        all_z.extend(z.flatten())

    return np.array(all_x), np.array(all_y), np.array(all_z)


def extract_parameter_values(radar_sweeps, dataset_key):
    """Extract parameter values from all radar sweeps."""
    
    all_payload = []

    for sweep in radar_sweeps:
        payload = sweep[dataset_key].values
        all_payload.extend(payload.flatten())

    return np.array(all_payload)