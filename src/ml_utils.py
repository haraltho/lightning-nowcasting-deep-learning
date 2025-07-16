import os
import h5py
import numpy as np

def get_file_splits(radar_dir, lightning_dir, train_ratio=0.8):
    """Split radar and lightning data files into training and validation sets."""

    radar_files = sorted([f for f in os.listdir(radar_dir) if f.endswith('.h5')])
    lightning_files = sorted([f for f in os.listdir(lightning_dir) if f.endswith('.h5')])
    
    n_files = len(radar_files)
    n_train = int(n_files * train_ratio)
    n_val = n_files - n_train

    train_radar = radar_files[:n_train]
    train_lightning = lightning_files[:n_train]
    test_radar = radar_files[n_train:]
    test_lightning = lightning_files[n_train:]

    return train_radar, train_lightning, test_radar, test_lightning


def load_data_to_tensors(radar_files, lightning_files, radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type):
    """
    Load radar and lightning data into tensors for ML training.
    
    Returns
    -------
    X : numpy.ndarray
        Radar features with shape [n_samples, n_lat, n_lon, n_altitudes, n_parameters]
        Grid indexing: [lat_index, lon_index] = [North-South, East-West]
    y : numpy.ndarray  
        Lightning targets with shape [n_samples, n_lat, n_lon]
        Grid indexing: [lat_index, lon_index] = [North-South, East-West]
    """

    # [height, width, altitude, parameters]
    # [10, 10, 1, 4]

    X = []
    y = []


    for radar_file, lightning_file in zip(radar_files, lightning_files):

        radar_path     = os.path.join(radar_dir, radar_file)
        lightning_path = os.path.join(lightning_dir, lightning_file)

        with h5py.File(radar_path, 'r') as radar_h5:
            with h5py.File(lightning_path, 'r') as lightning_h5:

                leadtime_group = f"lightning_{leadtime}min_leadtime"

                for timestamp in radar_h5.keys():
                    # Radar data
                    radar_params = []
                    for param in parameters:
                        param_data = radar_h5[timestamp][param][:, :, :n_altitudes]
                        radar_params.append(param_data)
                    radar_params = np.stack(radar_params, axis=-1)
                    X.append(radar_params)

                    # Lightning data
                    lightning_sample = lightning_h5[leadtime_group][timestamp][lightning_type][:]
                    y.append(lightning_sample)

    return np.array(X), np.array(y)