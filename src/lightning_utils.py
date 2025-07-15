from datetime import timedelta
import global_variables as global_vars
import local_variables as local_vars

import numpy as np
import pandas as pd
import requests
import h5py
import io
import os


def call_lightning_api(start_time, end_time, grid):
    """
    Fetch lightning data from FROST API for a given time period and spatial grid.

    Creates a bounding box from the grid coordinates and retrieves all lightning 
    strikes within that area and time window from the Norwegian Meteorological 
    Institute's FROST lightning API.

    Parameters
    ----------
    start_time : datetime
        Start of the time period to fetch lightning data
    end_time : datetime  
        End of the time period to fetch lightning data
    grid : numpy npz file
        Loaded grid file containing 'grid_lats' and 'grid_lons' arrays
        defining the spatial boundaries

    Returns
    -------
    pandas.DataFrame
        Lightning data with columns including lat, lon, timestamp components,
        peak_current, cloud type (0=cloud-to-ground, 1=intracloud), and
        various detection metadata

    Notes
    -----
    Requires FROST_CLIENT_ID in local_vars and FROST_ENDPOINT in global_vars.
    """

    # Create box around radar defined by the grid
    min_lat = np.min(grid["grid_lats"])
    max_lat = np.max(grid["grid_lats"])
    min_lon = np.min(grid["grid_lons"])
    max_lon = np.max(grid["grid_lons"])
    polygon = f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"

    # API parameters
    client_id = local_vars.FROST_CLIENT_ID
    endpoint = global_vars.FROST_ENDPOINT

    # Convert datetime to API string
    time_range = f"{start_time.strftime('%Y-%m-%dT%H:%M:%SZ')}/{end_time.strftime('%Y-%m-%dT%H:%M:%SZ')}"

    columns = ['year', 'month', 'day', 'hour', 'minute', 'second', 'nanoseconds',
            'lat', 'lon', 'peak_current', 'multi', 'nsens', 'dof', 'angle',
            'major', 'minor', 'chi2', 'rt', 'ptz', 'mrr', 'cloud', 'aI', 'sI', 'tI']

    params = {
            'referencetime': time_range,
            'geometry': polygon
        }

    data = requests.get(endpoint, params=params, auth=(client_id, ''))

    lightning_data = pd.read_csv(io.StringIO(data.text), sep=' ', names=columns)
    lightning_data.reset_index()
    return lightning_data


def lightning_to_grid(lightning_data, grid, time_segments, lead_time, time_window):
    lightning_data['datetime'] = pd.to_datetime(lightning_data[['year', 'month', 'day', 'hour', 'minute', 'second']])

    """
    Convert lightning strikes to gridded counts for multiple time segments.

    For each time segment, counts lightning strikes occurring within a future 
    time window and bins them onto a spatial grid by type (cloud-to-ground 
    and intracloud).

    Parameters
    ----------
    lightning_data : pandas.DataFrame
        Lightning data with columns: year, month, day, hour, minute, second, lat, lon, cloud, etc.
    grid : dict
        Grid definition with keys: x_centers_m, y_centers_m, grid_lats, grid_lons, etc.
    time_segments : list of datetime objects
        Reference times for which to create lightning targets
    lead_time : int
        Minutes ahead to start counting lightning (e.g., 30)
    time_window : int
        Duration in minutes to count lightning (e.g., 10)

    Returns
    -------
    dict
        Lightning grids keyed by timestamp ('08h00', '08h10', etc.) with 
        'cloud_to_ground', 'intracloud', and 'total' count arrays
    """

    # Load grid parameters
    n_x = len(grid["x_centers_m"])
    n_y = len(grid["y_centers_m"])
    latitudes  = grid["grid_lats"]
    longitudes = grid["grid_lons"]

    lightning_grids = {}

    for reference_time in time_segments:

        nowcast_start = reference_time + timedelta(minutes=lead_time)
        nowcast_end   = nowcast_start  + timedelta(minutes=time_window)

        # Filter lighting data for time window
        mask = (lightning_data['datetime'] >= nowcast_start) & (lightning_data['datetime'] < nowcast_end)
        lightning_filtered = lightning_data[mask]

        # Place lightnings onto grid
        cg_grid = np.zeros((n_y, n_x)) # Cloud-to-ground strikes
        ic_grid = np.zeros((n_y, n_x)) # Intra-cloud strikes

        for _, strike in lightning_filtered.iterrows():
            lat_idx = np.digitize(strike["lat"], latitudes)  - 1
            lon_idx = np.digitize(strike["lon"], longitudes) - 1
            if 0 <= lat_idx < n_y and 0 <= lon_idx < n_x:
                if strike["cloud"]==0:
                    cg_grid[lat_idx, lon_idx] += 1
                else:
                    ic_grid[lat_idx, lon_idx] += 1

        time_stamp = reference_time.strftime("%Hh%M")
        lightning_grids[time_stamp] = {
            'cloud_to_ground': cg_grid,
            'intracloud': ic_grid,
            'total': ic_grid + cg_grid,
        }

    return lightning_grids


def save_lightning_targets(lightning_grids, output_dir, label, lead_time, time_window, grid):
    """
    Save lightning target grids to HDF5 file.

    Parameters
    ----------
    lightning_grids : dict
        Lightning grids keyed by timestamp with cloud_to_ground, intracloud, total arrays
    output_dir : str
        Directory to save the file
    label : str
        Date label for filename (e.g., '2024-06-01')
    lead_time : int
        Lead time in minutes
    time_window : int
        Time window duration in minutes
    grid : dict
        Grid metadata for file attributes

    Returns
    -------
    None
        Saves targets_{label}.h5 file with lightning grids and metadata
    """

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}targets_{label}.h5"

    with h5py.File(filename, "w") as f:
        for timestamp, grids in lightning_grids.items():
            group = f.create_group(f"lightning_{lead_time}min_leadtime/{timestamp}")
            group.create_dataset('cloud_to_ground', data=grids['cloud_to_ground'])
            group.create_dataset('intracloud', data=grids['intracloud'])
            group.create_dataset('total', data=grids['total'])

            f.attrs['lead_time'] = lead_time
            f.attrs['time_window'] = time_window
            f.attrs['grid_spacing_m'] = grid['cell_size_m']
            f.attrs['radar_lat'] = grid['radar_lat']
            f.attrs['radar_lon'] = grid['radar_lon']

    print(f"Saved {filename}")