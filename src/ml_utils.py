import os
import h5py
import numpy as np
import tensorflow as tf


def get_file_splits(radar_dir, lightning_dir, train_ratio=0.7):
    """Split radar and lightning data files into training, validation and test set."""

    radar_files = sorted([f for f in os.listdir(radar_dir) if f.endswith('.h5')])
    lightning_files = sorted([f for f in os.listdir(lightning_dir) if f.endswith('.h5')])
    
    n_files = len(radar_files)
    n_train = int(n_files * train_ratio)
    n_val = int((n_files - n_train) / 2)
    n_test = n_files - n_train - n_val
    
    train_radar     = radar_files[:n_train]
    train_lightning = lightning_files[:n_train]
    validation_radar     = radar_files[n_train:n_train+n_val]
    validation_lightning = lightning_files[n_train:n_train+n_val]
    test_radar     = radar_files[n_train+n_val:]
    test_lightning = lightning_files[n_train+n_val:]

    return train_radar, train_lightning, validation_radar, validation_lightning, test_radar, test_lightning


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


def create_lightning_cnn(input_shape=(10, 10, 1, 4)):

    n_channels = input_shape[2] * input_shape[3]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((10, 10, n_channels)),  # Remove altitude dimension
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'),  # sigmoid for binary
        tf.keras.layers.Reshape((10, 10))
    ])
    return model


def calculate_csi(y_true, y_pred, threshold=0.5):
    """Calculate Critical Success Index"""

    y_true_binary = (y_true > 0).astype(int).flatten()
    y_pred_binary = (y_pred > threshold).astype(int).flatten()
    
    hits = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    misses = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    false_alarms = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0

    return csi


def evaluate_threshold(y_true, y_pred, n_thresholds=20):
    """Compute CSI for multiple threshold values"""

    max_pred = np.max(y_pred)

    thresholds = np.linspace(0.001, max_pred, n_thresholds)

    results = []
    for threshold in thresholds:
        csi = calculate_csi(y_true, y_pred, threshold=threshold)
        results.append((threshold, csi))

    return results