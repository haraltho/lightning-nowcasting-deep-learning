import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys


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


def load_data_to_tensors_temporal(radar_files, lightning_files, radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type, n_timesteps):
    """
    Load radar and lightning data into tensors for convLSTM training. 
    
    Returns
    -------
    X : numpy.ndarray
        Radar features with shape [n_samples, n_timesteps, n_lat, n_lon, n_altitudes, n_parameters]
            n_samples: number of target/feature pairs
            n_timesteps: number of timesteps per sample
            n_lat / n_lon: number of latitude/longitude steps
            n_altitudes: number of altitude layers
            n_parameters: number of parameters [dBZ, ...]
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
                timestamps = sorted(radar_h5.keys())

                for i in range(n_timesteps-1, len(timestamps)):
                    timestamps_slice = timestamps[i-n_timesteps+1:i+1]

                    temporal_sequence = []
                    for timestamp in timestamps_slice:
                        radar_params = []
                        for param in parameters:
                            param_data = radar_h5[timestamp][param][:, :, :n_altitudes]
                            radar_params.append(param_data)
                        radar_params = np.stack(radar_params, axis=-1)

                        temporal_sequence.append(radar_params)

                    temporal_sample = np.stack(temporal_sequence, axis=0)  # Shape: [n_timesteps, n_lat, n_lon, n_altitudes, n_parameters]
                    X.append(temporal_sample)

                    # Lightning data
                    lightning_sample = lightning_h5[leadtime_group][timestamps[i]][lightning_type][:]
                    y.append(lightning_sample)

    return np.array(X), np.array(y)


def get_shuffled_time_splits(radar_dir, lightning_dir, n_timesteps, train_ratio=0.7):
    
    radar_files = sorted([f for f in os.listdir(radar_dir) if f.endswith('.h5')])
    samples = []

    for file in radar_files:
        path = os.path.join(radar_dir, file)
        with h5py.File(path, 'r') as radar_h5:
            timestamps = sorted(radar_h5.keys())
            for timestamp in timestamps:
                date = file[-13:-3]
                sample = (date, timestamp)
                samples.append(sample)

    grouped_samples = []

    for i in range(len(samples) - n_timesteps + 1):
        group = samples[i:i+n_timesteps]

        if all([sample[0]==group[0][0] for sample in group]):
            grouped_samples.append(group)

    print(grouped_samples[0])
    return None, None, None, None, None, None


def create_lightning_cnn(input_shape=(10, 10, 1, 4), initial_bias=None):

    n_channels = input_shape[2] * input_shape[3] # n_elevations * n_parameters

    if initial_bias is not None:
        bias_initializer = tf.keras.initializers.Constant(initial_bias) 
    else:
        bias_initializer = 'zeros'
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((10, 10, n_channels)),  # Remove altitude dimension
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',    padding='same'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',    padding='same'),
        tf.keras.layers.Conv2D(1, (1, 1),  activation='sigmoid', padding='same', 
                               bias_initializer=bias_initializer),  # sigmoid for binary
        tf.keras.layers.Reshape((10, 10))
    ])
    return model


def create_lightning_convLSTM2D(input_shape, initial_bias):

    if initial_bias is not None:
        bias_initializer = tf.keras.initializers.Constant(initial_bias)
    else:
        bias_initializer = 'zeros'


    # Calculate number of channels
    n_altitudes  = input_shape[3]
    n_parameters = input_shape[4]
    n_channels   = n_altitudes * n_parameters
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((input_shape[0], input_shape[1], input_shape[2], n_channels)),
        tf.keras.layers.ConvLSTM2D(32, (3, 3), activation='relu', padding='same', return_sequences=False),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same', bias_initializer=bias_initializer),  # sigmoid for binary
        tf.keras.layers.Reshape((10, 10))      
    ])

    return model


def create_lightning_convLSTM3D(input_shape, initial_bias):

    if initial_bias is not None:
        bias_initializer = tf.keras.initializers.Constant(initial_bias)
    else:
        bias_initializer = 'zeros'

    # ConvLSTM3D requires the data to be in the shape:
    # (time, depth, *spatial_dims, channels) = (n_timesteps, n_altitudes, n_lat, n_lon, n_parameters)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.ConvLSTM3D(32, (3,3,3), activation='relu', padding='same', return_sequences=False),
        tf.keras.layers.Conv3D(16, (3,3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv3D(1, (input_shape[1], 1, 1), activation='sigmoid', padding='valid', bias_initializer=bias_initializer),
        tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1)),  # Remove altitude dim
        tf.keras.layers.Reshape((input_shape[2], input_shape[3]))  # Final (lat, lon)
    ])



    return model


def calculate_csi(y_true, y_pred, threshold=0.5):
    """Calculate Critical Success Index"""

    y_true_binary = (y_true > threshold).astype(int).flatten()
    y_pred_binary = (y_pred > threshold).astype(int).flatten()
    
    hits         = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    misses       = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    false_alarms = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0

    return csi


def evaluate_threshold(y_true, y_pred, n_thresholds=20):
    """Compute CSI for multiple threshold values"""

    max_pred = np.max(y_pred)

    thresholds = np.linspace(0.001, max_pred, n_thresholds)

    print('\n-- EVALUATE THRESHOLD --\nthreshold \t csi\n ' + 30*"-")

    results = []
    for threshold in thresholds:
        csi = calculate_csi(y_true, y_pred, threshold=threshold)
        results.append((threshold, csi))
        print(f"{threshold:.4f}\t\t{csi:.6f}")

    return results


def print_detailed_results(y_true, y_pred):

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    hits              = np.sum((y_true_flat==1) & (y_pred_flat==1))
    misses            = np.sum((y_true_flat==1) & (y_pred_flat==0))
    false_alarms      = np.sum((y_true_flat==0) & (y_pred_flat==1))
    correct_negatives = np.sum((y_true_flat==0) & (y_pred_flat==0))

    # Calculate metrics
    # Calculate metrics
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
    precision = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
    recall = hits / (hits + misses) if (hits + misses) > 0 else 0
    accuracy = (hits + correct_negatives) / len(y_true_flat)

    print(f"\n  Hits: {hits}, Misses: {misses}, False Alarms: {false_alarms}, Correct Negatives: {correct_negatives}")
    print(f"  CSI: {csi:.4f}")
    print(f"  Precision: {precision*100:.2f}%\t When lightning predicted, how often was it right")
    print(f"  Recall: {recall*100:.2f}%\t How many lightning events were caught")
    print(f"  Accuracy: {accuracy*100:.2f}%")


def visualize_results(X_test, y_test, y_pred, dir):

    output_dir = dir + "results/"
    os.makedirs(output_dir, exist_ok=True)

    n_samples = X_test.shape[0]

    for i in range(n_samples):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(X_test[i,0,:,:,0,0], cmap="seismic", vmin=-60, vmax=60)
        axes[0].set_title('Radar dBZ')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        im2 = axes[1].imshow(y_test[i,:,:], cmap="Reds", vmin=0, vmax=1)
        axes[1].set_title('True Lightning')

        im3 = axes[2].imshow(y_pred[i,:,:], cmap="Reds", vmin=0, vmax=1)
        axes[2].set_title('Predicted Lightning')

        plt.tight_layout()
        plt.savefig(f"{output_dir}prediction_{str(i).zfill(4)}.png", dpi=150, bbox_inches='tight')
        plt.close()


def compute_normalization_parameters(X, n_param):

    means = np.zeros(n_param)
    stds  = np.zeros(n_param)

    for i in range(n_param):
        X_i = X[:,0,:,:,:,i]
        means[i] = np.nanmean(X_i)
        stds[i]  = np.nanstd(X_i)

    return means, stds


def normalize_and_preprocess(X, means, stdevs, n_param):

    # shape(X): (n_samples, n_timesteps, n_lat, n_lon, n_alt, n_param)

    X = X.copy()

    for i in range(n_param):
        X_i    = X[:,:,:,:,:,i]
        mean   = means[i]
        stdev  = stdevs[i]
        X_i    = (X_i - mean) / stdev

        X[:,:,:,:,:,i] = X_i

    mask = ~np.isnan(X)
    X[~mask] = 0

    return X, mask


def visualize_timeline(y_test, y_pred, run_dir):

    output_dir = run_dir + "results/"
    os.makedirs(output_dir, exist_ok=True)

    lightning_true = np.sum(y_test, axis=(1,2))
    lightning_pred = np.sum(y_pred, axis=(1,2))

    fig, axes = plt.subplots(1,1, figsize=(15,5))
    axes.plot(lightning_true, label="true")
    axes.plot(lightning_pred, label="predicted")
    axes.set_xlabel("sample")
    axes.set_ylabel("number of lightnings in grid")
    axes.legend()
    fig.savefig(f"{output_dir}time_correlation.png", dpi=150, bbox_inches="tight")

