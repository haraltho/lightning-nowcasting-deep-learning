import os
import h5py
import numpy as np
import tensorflow as tf


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


def create_weighted_loss(lightning_weight):

    # Calculate weight once
    

    print(f"\nUsing lightning weight: {lightning_weight:.1f}")

    def weighted_mse(y_true, y_pred):
        weights = tf.where(y_true > 0, lightning_weight, 1.0)
        squared_diff = tf.square(y_true - y_pred)
        return tf.reduce_mean(weights * squared_diff)
    
    return weighted_mse


def create_lightning_cnn(input_shape=(10, 10, 1, 4)):
    layers = tf.keras.layers

    # Boilderplate CNN model
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((10, 10, 4)),  # Remove altitude dimension (10,10,1,4) → (10,10,4)
        
        # Simple 2D convolutions to capture spatial patterns
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
        
        # Final prediction
        layers.Conv2D(1, kernel_size=(1, 1), activation='linear', padding='same'),
        layers.Reshape((10, 10))  # Output: (10, 10) lightning predictions
    ])
    return model
