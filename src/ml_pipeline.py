import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import importlib
import os
import random
from sklearn.metrics import average_precision_score, roc_auc_score
import sys

import ml_utils
importlib.reload(ml_utils)

# Set seed number for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)

# Configurations
run_dir = "../data/processed_data/run_6_leadtime15/"
radar_dir     = run_dir + "radar/"
lightning_dir = run_dir + "lightning/"
# parameters    = ['dBZ', 'ZDR', 'KDP', 'RhoHV']
parameters    = ['dBZ']
n_altitudes   = 20  # Using only lowest altitude for simplicity
leadtime      = 15
lightning_type = "total" # "total" or "cloud_to_ground" or "intracloud"
n_timesteps = 6  # Number of timesteps for convLSTM
model_type    = "convlstm2d" # "convlstm2d" or "convlstm3d"

# Step 1: Split data into training days, validation days and test days
# print("\nSplitting data into training and test sets...")
# train_radar, train_lightning, validation_radar, validation_lightning, test_radar, test_lightning = ml_utils.get_file_splits(radar_dir, lightning_dir)


# Step 2: Load h5 file and return tensors
# print("\nLoading data into tensors...")
# # shape(X) = [n_samples, n_timesteps, n_lat, n_lon, n_altitudes, n_parameters]
# X_train, y_train = ml_utils.load_data_to_tensors_temporal(train_radar, train_lightning, radar_dir, 
#                                                  lightning_dir, n_altitudes, parameters, 
#                                                  leadtime, lightning_type, n_timesteps)
# X_val,   y_val   = ml_utils.load_data_to_tensors_temporal(validation_radar, validation_lightning, 
#                                                  radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type, n_timesteps)
# X_test , y_test  = ml_utils.load_data_to_tensors_temporal(test_radar,  test_lightning,  radar_dir, 
#                                                  lightning_dir, n_altitudes, parameters, 
#                                                  leadtime, lightning_type, n_timesteps)


# Step 1: Split data into training days, validation days and test days
print("\nSplitting data into training and test sets...")
train_samples, validation_samples, test_samples = ml_utils.get_shuffled_time_splits(radar_dir, n_timesteps)
sys.exit("after loading into tensors")

# Convert targets to binary
y_train_binary = (y_train>0).astype(float)
y_val_binary   = (y_val>0).astype(float)
y_test_binary  = (y_test>0).astype(float)

print(f"Lightning fraction in train: {np.mean(y_train_binary)*100:.2f}%")
print(f"Lightning fraction in val:   {np.mean(y_val_binary)*100:.2f}%")
print(f"Lightning fraction in test:  {np.mean(y_test_binary)*100:.2f}%")


# Normalizing data: Fill nan's, produce mask
print("\nNormalizing the data...")
means, stdevs = ml_utils.compute_normalization_parameters(X_train, len(parameters))
X_train_normalized, _ = ml_utils.normalize_and_preprocess(X_train, means, stdevs, len(parameters))
X_val_normalized,   _ = ml_utils.normalize_and_preprocess(X_val,   means, stdevs, len(parameters))
X_test_normalized,  _ = ml_utils.normalize_and_preprocess(X_test,  means, stdevs, len(parameters))

print("\nBefore normalization: X_train mean and std per parameter:")
print("mean: ", np.nanmean(X_train, axis=(0,1,2,3,4)))  
print("std: ",  np.nanstd(X_train,  axis=(0,1,2,3,4)))  

print("\nAfter normalization: X_train mean and std per parameter:")
print(np.nanmean(X_train_normalized, axis=(0,1,2,3,4)))  
print(np.nanstd(X_train_normalized,  axis=(0,1,2,3,4)))  


# Step 4: Deal with class imbalance
neg_count = np.sum(y_train_binary == 0)
pos_count = np.sum(y_train_binary == 1)
initial_bias = np.log(pos_count / neg_count)
print(f"\nInitial class imbalance: {neg_count} negatives, {pos_count} positives")
print(f"Initial bias for loss function: {initial_bias:.4f}\n")


# Step 5: Create a simple CNN model

if model_type=="convlstm2d":
    model = ml_utils.create_lightning_convLSTM2D(np.shape(X_test_normalized)[1:], initial_bias)
elif model_type=="convlstm3d":
    # Transpose to from
    # [n_samples, n_timesteps, n_lat, n_lon, n_altitudes, n_parameters]
    # to
    # [n_samples, n_timesteps, n_altitudes, n_lat, n_lon, n_parameters]
    # since this shape is required by ConvLSTM3D
    transpose_indices = (0, 1, 4, 2, 3, 5)
    X_train_normalized = np.transpose(X_train_normalized, transpose_indices)
    X_val_normalized   = np.transpose(X_val_normalized,   transpose_indices)
    X_test_normalized  = np.transpose(X_test_normalized,  transpose_indices)
    model = ml_utils.create_lightning_convLSTM3D(np.shape(X_test_normalized)[1:], initial_bias)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['precision', 'recall']
)

# Step 6: Train the model
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

history = model.fit(
        X_train_normalized,
        y_train_binary,
        epochs=200,
        batch_size=32,
        validation_data=(X_val_normalized, y_val_binary),
        callbacks=[early_stopping]
)

# Step 7: Evaluate the model
print("\n" + "="*50)
print("MODEL EVALUATION")
print("\n" + "="*50)

# Get predictions
y_pred = model.predict(X_test_normalized)

# Predictions give probablities. Convert back to binary by exploring different thresholds.
threshold_csi = ml_utils.evaluate_threshold(y_test_binary, y_pred)

# Find the threshold with highest CSI
best_threshold, best_csi = max(threshold_csi, key=lambda x: x[1])
print(f"\nBest threshold: {best_threshold:.4f}")
print(f"Best CSI: {best_csi:.4f}")

# Make final binary predictions
y_pred_binary = (y_pred > best_threshold).astype(int)

# Generate detailed output
ml_utils.print_detailed_results(y_test_binary, y_pred_binary)

# Calculate threshold-independent metrics
auc_pr  = average_precision_score(y_test_binary.flatten(), y_pred.flatten())
auc_roc = roc_auc_score(y_test_binary.flatten(), y_pred.flatten())

print(f"\nAUC-PR: {auc_pr:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

# Plot X_true, y_true and y_pred
print("\n-- PLOTTING RESULTS --")
# ml_utils.visualize_results(X_test, y_test_binary, y_pred_binary, run_dir)

# Plot number of lightnings as function of time
ml_utils.visualize_timeline(y_test_binary, y_pred_binary, run_dir)

print("\n-- FINISHED --")

