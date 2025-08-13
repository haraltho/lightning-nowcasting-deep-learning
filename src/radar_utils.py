from datetime import timedelta
import glob
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

import h5py
import os

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
        
    print(f"No radar file found for reference time or up to 30min earlier.")
    return None


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


def interpolate_to_grid(x, y, z, values, grid):
    """
    Interpolate radar data onto regular 3D grid using nearest neighbor.
    
    Returns
    -------
    numpy.ndarray
        Interpolated values with shape (n_y, n_x, n_z)
    """

    x_m = grid['x_centers_m']
    y_m = grid['y_centers_m']
    z_m = grid['z_levels_m']

    # Create 3D grid
    x_grid, y_grid, z_grid = np.meshgrid(x_m, y_m, z_m)

    # Only use valid data
    valid = ~np.isnan(values)
    x_valid = x[valid]
    y_valid = y[valid]
    z_valid = z[valid]
    values_valid = values[valid]

    # Do the interpolation
    grid_values = griddata(points=np.column_stack([x_valid, y_valid, z_valid]),
                           values=values_valid,
                           xi=np.column_stack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()]),
                           method="nearest")

    values_interpolated = grid_values.reshape(x_grid.shape)
    return values_interpolated


def k_nearest_neighbors_anisotropic(x, y, z, values, grid, k):
    """
    Interpolate radar data to regular 3D grid using anisotropic k-nearest neighbors.
    
    Uses vertical scaling to account for different grid spacing in horizontal vs. 
    vertical directions.
    
    Parameters
    ----------
    x, y, z : array_like
        Radar measurement coordinates in meters  
    values : array_like
        Radar values to interpolate
    grid : dict
        Grid definition containing 'x_centers_m', 'y_centers_m', 'z_levels_m'
    k : int, default 4
        Number of nearest neighbors to find

        
    Returns
    -------
    ndarray
        Interpolated values with shape (n_y, n_x, n_z), NaN where insufficient neighbors
    """

    x_m = grid['x_centers_m']
    y_m = grid['y_centers_m']
    z_m = grid['z_levels_m']

    radius = grid['cell_size_m'] / 2
    max_distance = np.sqrt(radius**2 + radius**2 + radius**2)

    vertical_scale = (x_m[1] - x_m[0]) / (z_m[1] - z_m[0])

    # Create 3D grid to interpolate to
    x_grid, y_grid, z_grid = np.meshgrid(x_m, y_m, z_m)

    # Flatten grid
    grid_points = np.column_stack([
        x_grid.flatten(),
        y_grid.flatten(),
        z_grid.flatten() * vertical_scale  # Scale vertical coordinate
    ])

    # Only use valid data
    valid = ~np.isnan(values)
    x_valid = x[valid]
    y_valid = y[valid]
    z_valid = z[valid]
    dbzh_valid = values[valid]

    # Rescale z before building the KDTree
    points = np.column_stack([
        x_valid,
        y_valid,
        z_valid * vertical_scale
    ])
    tree = cKDTree(points)

    # Find k nearest neighbors for each voxel
    distances, indices = tree.query(grid_points, k=k)

    # Initialize result array with NaNs
    interpolated = np.full(len(grid_points), np.nan)

    # Loop through each grid point (voxel)
    for i in range(len(grid_points)):
        neighbor_idxs = indices[i]
        d = distances[i]

        # Select only neighbors within max_distance
        within_mask = d < max_distance

        if np.any(within_mask):
            valid_values = dbzh_valid[neighbor_idxs[within_mask]]
            interpolated[i] = np.mean(valid_values)  

    # Reshape to grid
    interpolated_final = interpolated.reshape(x_grid.shape)

    return interpolated_final


def save_radar_features(daily_features, output_dir, date_label, grid):
    """
    Save radar features for a single day to an HDF5 file.
    
    Parameters
    ----------
    daily_features : dict
        Dictionary with time segments as keys and feature arrays as values
    output_dir : str
        Directory to save the HDF5 file
    date_label : str
        Date label for the file name (e.g. '2023-10-01')
    grid : dict
        Grid definition for metadata
    
    Returns
    -------
    None
    """
    
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}features_{date_label}.h5"

    with h5py.File(filename, "w") as f:
        for timestamp, time_features in daily_features.items():
            group = f.create_group(timestamp)
            for param, param_grid in time_features.items():
                group.create_dataset(param, data=param_grid)
        
        f.attrs['grid_spacing_m'] = grid['cell_size_m']
        f.attrs['radar_lat'] = grid['radar_lat']
        f.attrs['radar_lon'] = grid['radar_lon']

    print(f"Saved {filename}")