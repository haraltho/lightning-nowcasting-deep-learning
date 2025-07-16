import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import importlib

import ml_utils
importlib.reload(ml_utils)

# Data directories
data_dir = "../data/processed_data/"
radar_dir = data_dir + "radar/"
lightning_dir = data_dir + "lightning/"

# Step 1: Split data into training days and validation days
train_radar, train_lightning, test_radar, test_lightning = ml_utils.get_file_splits(radar_dir, lightning_dir)


