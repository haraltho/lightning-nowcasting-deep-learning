import os



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