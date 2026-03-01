import os
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import random


def get_file_splits(radar_dir, lightning_dir, train_ratio=0.7):
    """
    Split radar and lightning data files into training, validation and test set.
    
    Parameters
    ----------
    radar_dir : str
        Directory where the processed radar data are stored.
    lightning_dir : str
        Directory where the processed lightning data are stored.
    train_ratio : float
        Proportion of data that is used for training. The rest
        is used for testing and validating.

    Returns
    -------
    train_radar, train_lightning : list of str
        Lists of radar and corresponding lightning files used for training.
    validation_radar, validation_lightning : list of str
        Lists of radar and corresponding lightning files used for validation.
    test_radar, test_lightning : list of str
        Lists of radar and corresponding lightning files used for testing.
    """

    radar_files = sorted([f for f in os.listdir(radar_dir) if f.endswith('.h5')])
    lightning_files = sorted([f for f in os.listdir(lightning_dir) if f.endswith('.h5')])
    
    n_files = len(radar_files)
    n_train = int(n_files * train_ratio)
    n_val = int((n_files - n_train) / 2)
    
    train_radar     = radar_files[:n_train]
    train_lightning = lightning_files[:n_train]
    validation_radar     = radar_files[n_train:n_train+n_val]
    validation_lightning = lightning_files[n_train:n_train+n_val]
    test_radar     = radar_files[n_train+n_val:]
    test_lightning = lightning_files[n_train+n_val:]

    return train_radar, train_lightning, validation_radar, validation_lightning, test_radar, test_lightning


def get_file_splits_shuffled_by_day(radar_dir, lightning_dir, train_ratio=0.7, holdout_year=None):
    """
    Split the radar and lightning files by day with an optional holdout year.

    Ensures that the radar and lightning files correspond by matching dates. Optionally
    collects all files from a given year as a holdout set. The remaining days are split
    into training, validation, and test sets. Validation samples are taken as a 
    consecutive block, while the training and test samples are shuffled.

    Parameters
    ----------
    radar_dir : str
        Directory containing the processed radar feature files.
    lightning_dir : str
        Directory containing the processed lightning target files.
    train_ratio : float, default=0.7
        Proportion of non-holdout data used for training.
    holdout_year : int or None
        Year to reserve entirely as a holdout set (e.g. 2025). 
        If None, all years are included in the split.
    
    Returns
    -------
    radar_train, lightning_train : list of str
        Lists containing the training files.
    radar_val, lightning_val : list of str
        Lists containing the validation files.
    radar_test, lightning_test : list of str
        Lists containing the test files.
    radar_holdout, lightning_holdout : list of str
        Lists containing the holdout files. 
        Empty if holdout year is None.
    """
    
    radar_files = sorted([f for f in os.listdir(radar_dir) if f.endswith('.h5')])
    lightning_files = sorted([f for f in os.listdir(lightning_dir) if f.endswith('.h5')])

    # Make sure that radar and lightning files correspond. Extract date from lightning and radar files, compute intersection
    def extract_date(filename):
        return filename.split('_')[-1].replace('.h5', '')
    
    radar_dates     = {extract_date(f): f for f in radar_files}
    lightning_dates = {extract_date(f): f for f in lightning_files}
    common_dates = radar_dates.keys() & lightning_dates.keys()
    assert len(common_dates) > 0, "No common dates found between radar and lightning files"

    radar_files =     [radar_dates[date] for date in sorted(common_dates)]
    lightning_files = [lightning_dates[date] for date in sorted(common_dates)]

    # Pop holdout year files to separate list
    if holdout_year is not None:
        radar_holdout     = [f for f in radar_files     if extract_date(f).startswith(str(holdout_year))]
        lightning_holdout = [f for f in lightning_files if extract_date(f).startswith(str(holdout_year))]

        radar_files     = [f for f in radar_files     if not extract_date(f).startswith(str(holdout_year))]
        lightning_files = [f for f in lightning_files if not extract_date(f).startswith(str(holdout_year))]

    n_files = len(radar_files)
    n_train = int(n_files * train_ratio)
    n_val = int((n_files - n_train) / 2)
    n_test = n_files - n_train - n_val


    # Extract consecutive validation samples from a random position
    val_start = np.random.randint(0, n_files - n_val)
    lightning_val = lightning_files[val_start:val_start+n_val]
    radar_val = radar_files[val_start:val_start+n_val]
    del lightning_files[val_start:val_start+n_val]
    del radar_files[val_start:val_start+n_val]

    # Shuffle remaining samples and extract train and test samples
    combined = list(zip(radar_files, lightning_files))
    random.shuffle(combined)
    radar_files, lightning_files = zip(*combined)
    radar_files = list(radar_files)
    lightning_files = list(lightning_files)
    radar_train = radar_files[:n_train]
    lightning_train = lightning_files[:n_train]
    radar_test = radar_files[n_train:]
    lightning_test = lightning_files[n_train:]

    return radar_train, lightning_train, radar_val, lightning_val, radar_test, lightning_test, radar_holdout, lightning_holdout


def load_data_to_tensors(radar_files, lightning_files, radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type):
    """
    Load radar feature and lightning target files into NumPy tensors.

    For each pair of radar and lightning HDF5 files, all timestamps are
    iterated over and stacked into feature and target arrays suitable for 
    2D CNN training (single-timestep samples).

    Parameters
    ----------
    radar_files : list of str
        Filenames of processed radar feature files.
    lightning_files : list of str
        Filenames of processed lightning target files.
    radar_dir : str
        Directory containing the radar feature files.
    lightning_dir : str
        Directory containing the lightning target files.
    n_altitudes : int
        Number of altitude layers to include.
    parameters : list of str
        Radar parameters to load (e.g. ['dBZ', 'ZDR])
    leadtime : int
        Lightning prediction lead time in minutes.
    lightning_type : str
        The type of lightning target to load ('total', 'cloud_to_ground',
        or 'intracloud')
    
    Returns
    -------
    X : numpy.ndarray
        Radar tensor with shape [n_samples, n_lat, n_lon, n_altitudes, n_parameters]
    y : numpy.ndarray  
        Lightning tensors with shape [n_samples, n_lat, n_lon]
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
    Load radar feature and lightning target files into temporal tensors.

    Creates samples consisting of consecutive radar timesteps for use in
    ConvLSTM-based models. For each valid sequence of length 'n_timesteps',
    the radar data are stacked into a 5D tensor and paired with the corresponding
    lightning target at the final timestamp of the sequence.

    Parameters
    ----------
    radar_files : list of str
        Filenames of the processed radar feature files.
    lightning_files : list of str
        Filenames of the processed lightning target files.
    radar_dir : str
        Directory containing the radar feature files.
    lightning_dir : str
        Directory containing the lightning target files.
    n_altitudes : int
        Number of altitude layers to include.
    parameters : list of str
        Radar parameters to load (e.g., ['dBZ', 'ZDR']).
    leadtime : int
        Lightning prediction lead time in minutes.
    lightning_type : str
        Type of lightning target to load ('total', 'cloud_to_ground',
        or 'intracloud').
    n_timesteps : int
        Number of consecutive timesteps per sample.
    
    Returns
    -------
    X : numpy.ndarray
        Radar tensors with shape [n_samples, n_timesteps, n_lat, n_lon, n_altitudes, n_parameters]
    y : numpy.ndarray  
        Lightning tensors with shape [n_samples, n_lat, n_lon]
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

                        # Radar data
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


def get_shuffled_time_splits(radar_dir, n_timesteps, train_ratio=0.7):
    """
    Create shuffled temporal data splits.
    
    Groups consecutive timesteps into sequences, extracts consecutive validation 
    samples from a random position, then shuffles remaining sequences for 
    train/test splits. This approach preserves temporal structure within 
    sequences.
    
    Parameters
    ----------
    radar_dir : str
        Directory containing radar HDF5 files
    n_timesteps : int
        Number of consecutive timesteps per sequence (e.g., 6)
    train_ratio : float, default=0.7
        Proportion of data for training. Remaining data split equally 
        between validation and test sets.
        
    Returns
    -------
    train_groups : list
        List of training sequences
    val_groups : list  
        List of validation sequences
    test_groups : list
        List of test sequences
        
    Notes
    -----
    - Only creates sequences within the same day (no cross-day sequences)
    - Validation set is consecutive samples for temporal generalization testing
    - Train and test sets are shuffled to ensure representative sampling
    """
    
    radar_files = sorted([f for f in os.listdir(radar_dir) if f.endswith('.h5')])
    samples = []

    # Find all time samples
    for file in radar_files:
        path = os.path.join(radar_dir, file)
        with h5py.File(path, 'r') as radar_h5:
            timestamps = sorted(radar_h5.keys())
            for timestamp in timestamps:
                date = file[-13:-3]
                sample = (date, timestamp)
                samples.append(sample)


    # Group time samples
    grouped_samples = []
    for i in range(len(samples) - n_timesteps + 1):
        group = samples[i:i+n_timesteps]

        if all([sample[0]==group[0][0] for sample in group]):
            grouped_samples.append(group)

    # Define split sizes
    n_grouped_samples = len(grouped_samples)
    val_ratio  = (1-train_ratio)/2
    test_ratio = (1-train_ratio)/2
    n_val  = int(val_ratio  * n_grouped_samples)
    n_test = int(test_ratio * n_grouped_samples)
    n_train = n_grouped_samples - n_val - n_test

    # Extract consecutive validation samples from a random position
    val_start = np.random.randint(0, n_grouped_samples - n_val)
    val_groups = grouped_samples[val_start:val_start+n_val]
    del grouped_samples[val_start:val_start+n_val]

    # Shuffle remaining samples and group validation and test samples
    np.random.shuffle(grouped_samples)
    test_groups = grouped_samples[:n_test]
    train_groups = grouped_samples[n_test:]

    return train_groups, val_groups, test_groups


def load_shuffled_data_to_tensors(samples, radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type, n_timesteps):
    """
    Load grouped temporal samples into training tensors for ConvLSTM models.
    
    Loads radar and lightning data for grouped temporal sequences, creating
    4D radar tensors (time, lat, lon, altitude, parameters) and 2D lightning
    targets based on the final timestamp in each sequence.
    
    Parameters
    ----------
    samples : list
        List of temporal sequences from get_shuffled_time_splits().
        Each sequence contains n_timesteps tuples of (date, timestamp).
    radar_dir : str
        Directory containing radar feature HDF5 files
    lightning_dir : str  
        Directory containing lightning target HDF5 files
    n_altitudes : int
        Number of altitude levels to include from radar data
    parameters : list of str
        Radar parameters to load (e.g., ['dBZ', 'ZDR', 'KDP', 'RhoHV'])
    leadtime : int
        Lightning prediction lead time in minutes
    lightning_type : str
        Type of lightning to predict ('total', 'cloud_to_ground', 'intracloud')
    n_timesteps : int
        Number of timesteps per sequence (should match input sequences)
        
    Returns
    -------
    X : numpy.ndarray
        Radar features with shape [n_samples, n_timesteps, n_lat, n_lon, n_altitudes, n_parameters]
    y : numpy.ndarray
        Lightning targets with shape [n_samples, n_lat, n_lon]
        
    Notes
    -----
    Uses the last timestamp in each sequence as reference time for lightning prediction.
    """


    X = []
    y = []
    
    for sample_group in samples:
        date = sample_group[0][0]
        radar_file     = os.path.join(radar_dir,     f"features_{date}.h5")
        lightning_file = os.path.join(lightning_dir, f"targets_{date}.h5")

        with h5py.File(radar_file, 'r') as radar_h5:
            with h5py.File(lightning_file, 'r') as lightning_h5:

                # Radar data
                temporal_sequence = []
                for sample in sample_group:
                    timestamp = sample[1]
                    radar_params = []
                    for param in parameters:
                        param_data = radar_h5[timestamp][param][:, :, :n_altitudes]
                        radar_params.append(param_data)
                    radar_params = np.stack(radar_params, axis=-1)

                    temporal_sequence.append(radar_params)

                temporal_sample = np.stack(temporal_sequence, axis=0)
                X.append(temporal_sample)

                # Lightning data
                leadtime_group = f"lightning_{leadtime}min_leadtime"
                timestamp = [sample[1] for sample in sample_group][-1] # Use last timestamp as reference time for lightning prediction
                lightning_sample = lightning_h5[leadtime_group][timestamp][lightning_type][:]
                y.append(lightning_sample)


    return np.array(X), np.array(y)


def create_lightning_cnn(input_shape, initial_bias=None):
    """
    Build a 2D CNN model for lightning prediction with a single timestep.

    The input consists of one radar snapshot with shape (H, W, Z, C), where
    H and W spatial dimensions, Z is the number of altitude layers, and C is 
    the number of radar parameters. The altitude and parameter dimensions are 
    flattened into channels before applying 2D convolutions. 

    Parameters
    ----------
    input_shape : tuple
        Shape of a single radar sample (H, W, Z, C).
    initial_bias : float or None
        Optional bias initializer to account for class imbalance.

    Returns
    -------
    tf.keras.Model
        Compiled 2D CNN model with sigmoid output per grid cell.
    """

    H, W, Z, C = input_shape
    n_channels = Z * C

    if initial_bias is not None:
        bias_initializer = tf.keras.initializers.Constant(initial_bias) 
    else:
        bias_initializer = 'zeros'
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Reshape((H, W, n_channels)),  # Remove altitude dimension
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',    padding='same'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',    padding='same'),
        tf.keras.layers.Conv2D(1, (1, 1),  activation='sigmoid', padding='same', 
                               bias_initializer=bias_initializer),  # sigmoid for binary
        tf.keras.layers.Reshape((H, W))
    ])
    return model


def create_lightning_convLSTM2D(input_shape, initial_bias):
    """
    Build a ConvLSTM2D model for temporal lightning prediction.

    The input consists of a sequence of radar snapshots with shape 
    (T, H, W, Z, C), where T is the number of timesteps, H and W 
    are spatial dimensions, Z is the number of altitude layers, and 
    C is the number of radar parameters. The altitude and parameter
    dimensions are flattened into channels before applying ConvLSTM2D
    operations.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of a temporal radar sample (T, H, W, Z, C).
    initial_bias : float or None
        Optional bias initializer to account for class imbalance.

    Returns
    -------
    tf.keras.Model
        ConvLSTM2D model with sigmoid output per grid cell.
    """

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
    """
    Build a ConvLSTM3D model for temporal and spatial lightning prediction.

    The input consists of a sequence of radar snapshots with shape 
    (T, H, W, Z, C), where T is the number of timesteps, H and W 
    are spatial dimensions, Z is the number of altitude layers, and 
    C is the number of radar parameters. Convolutions are performed
    jointly in time and 3D space before reducing the altitude dimension
    to produce a 2D lightning map.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of a temporal radar sample (T, H, W, Z, C).
    initial_bias : float or None
        Optional bias initializer to account for class imbalance.

    Returns
    -------
    tf.keras.Model
        ConvLSTM3D model with sigmoid output per grid cell.
    """

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
    """
    Calculate the Critical Success Index (CSI) for binary predictions.

    Applies a threshold to continuous predictions, converts both true
    and predicted values to binary fields, and computes
    CSI = hits / (hits + misses + false alarms).

    Parameters
    ----------
    y_true : numpy.ndarray
        True lightning map
    y_pred : numpy.ndarray
        Predicted lightning probabilities.
    threshold : float, default=0.5
        Threshold used to convert predictions to binary values.

    Returns
    -------
    csi : float
        Critical Success Index value
    """

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
    """
    Print confusion matrix counts and derived metrics.

    Computes hits, misses, false alarms, and correct negatives from
    binary predictions, then prints CSI, precision, recall and
    accuracy.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        True lightning map (binary).
    y_pred : numpy.ndarray
        Predicted lightning map (binary).

    Returns
    -------
    csi, precision, recall, accuracy : float
    """

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

    return csi, precision, recall, accuracy


def visualize_results(X_test, y_test, y_pred, dir):
    """
    Save side-by-side visual comparison of radar, true lightning,
    and predicted lightning fields. 

    For each sample, generate a three-panel figure showing a radar
    snapshot, the true lightning map, and the predicted lightning 
    map. Images are written to an output directory.

    Parameters
    ----------
    X_test : numpy.ndarray
        Radar test tensor.
    y_test : numpy.ndarray
        True lightning tensor.
    y_pred : numpy.ndarray
        Predicted lightning tensor.
    dir : str
        Base directory where results will be stored. 

    Returns
    -------
    None
        Figures are saved ot disk.

    """

    output_dir = os.path.join(dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    n_samples = X_test.shape[0]

    for i in range(n_samples):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        radar_img = X_test[i, 0, :, :, 0, 0]
        im1 = axes[0].imshow(
            radar_img,
            cmap="RdBu_r",
            vmin=-60,
            vmax=60,
            interpolation="nearest",
        )
        axes[0].set_title("Radar (dBZ)", fontsize=11)
        cbar = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        cbar.set_label("dBZ")

        # true lightning panel
        truth = y_test[i, :, :]
        axes[1].imshow(
            truth,
            cmap="Oranges",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            alpha=0.95,
        )
        axes[1].set_title("True lightning", fontsize=11)

        # predicted lightning panel
        pred = y_pred[i, :, :]
        axes[2].imshow(
            pred,
            cmap="Oranges",
            vmin=0,
            vmax=1,
            interpolation="nearest",
            alpha=0.95,
        )

        axes[2].set_title("Predicted lightning", fontsize=11)

        # remove ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])


        fig.savefig(
            os.path.join(output_dir, f"prediction_{i:04d}.png"),
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def compute_normalization_parameters(X, n_param):
    """
    Compute normalization statistics per parameter.

    Calculates the mean and standard deviation for each radar
    parameter across all samples (using the first timestep),
    ignoring NaN values.

    Parameters
    ----------
    X : numpy.ndarray
        Radar tensor with shape
        (n_samples, n_timesteps, n_lat, n_lon, n_alt, n_param).
    n_param : int
        Number of radar parameters.

    Returns
    -------
    means : numpy.ndarray
        Mean value per radar parameter.
    stds : numpy.ndarray
        Standard deviation per radar parameter.
    """

    means = np.zeros(n_param)
    stds  = np.zeros(n_param)

    for i in range(n_param):
        X_i = X[:,0,:,:,:,i]
        means[i] = np.nanmean(X_i)
        stds[i]  = np.nanstd(X_i)

    return means, stds


def normalize_and_preprocess(X, means, stdevs, n_param):
    """
    Normalize the radar tensors and replace NaNz with zeros.

    Applies standardization per parameter using precomputed means
    and standard deviations. NaN values are replaced with zeros and 
    a boolean mask of valid entries is returned.
    
    Parameters
    ----------
    X : numpy.ndarray
        Radar tensor with shape
        (n_samples, n_timesteps, n_lat, n_lon, n_alt, n_param).
    means : numpy.ndarray
        Mean values per radar parameter.
    stdevs : numpy.ndarray
        Standard deviations per radar parameter.
    n_param : int
        Number of radar parameters.

    Returns
    -------
    X_norm : numpy.ndarray
        Normalized radar tensor.
    mask : numpy.ndarray
        Boolean mask indicating non-NaN entries.
    """

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
    """
    Plot total lightning counts inside the grid over time
    for true and predicted data.

    Parameters
    ----------
    y_test : numpy.ndarray
        True lightning tensor
    y_pred : numpy.ndarray
        Predicted lightning tensor.
    run_dir : str
        Directory where the results will be stored.

    Returns
    -------
    None
        Figure is saved to disk.
    """

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


def evaluate_holdout_year(model, radar_holdout, lightning_holdout,
                          radar_dir, lightning_dir, n_altitudes, 
                          parameters, leadtime, lightning_type,
                          n_timesteps, means, stdevs, best_threshold,
                          model_type, run_dir):
    """
    Evaluate holdout year and generate visualizations.
    """

    if len(radar_holdout) == 0:
        return
        
    print("\n-- EVALUATION OF HOLDOUT YEAR (2025) --")

    X_25, y_25 = load_data_to_tensors_temporal(
        radar_holdout, lightning_holdout, radar_dir, lightning_dir,
        n_altitudes, parameters, leadtime, lightning_type, n_timesteps)

    X_25_norm, _ = normalize_and_preprocess(X_25, means, stdevs, len(parameters))

    if model_type == "cnn2d":
        X_25_norm = X_25_norm[:, -1, ...]  
    elif model_type == "convlstm3d":
        X_25_norm = np.transpose(X_25_norm, (0, 1, 4, 2, 3, 5))

    y_25_pred = model.predict(X_25_norm, verbose=0)
    y_25_bin  = (y_25_pred > best_threshold).astype(int)

    y_25_binary = (y_25 > 0).astype(float)
    print_detailed_results(y_25_binary, y_25_bin)

    visualize_results(X_25, y_25_binary, y_25_bin, run_dir)
    visualize_timeline(y_25_binary, y_25_bin, run_dir)
    print(f"Saved holdout predictions to {os.path.join(run_dir, 'predictions_holdout.npz')}")