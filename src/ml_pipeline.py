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
run_dir = "../data/processed_data/run_5_knn100/"
radar_dir     = run_dir + "radar/"
lightning_dir = run_dir + "lightning/"
# parameters    = ['dBZ', 'ZDR', 'KDP', 'RhoHV']
parameters    = ['dBZ', 'ZDR']
n_altitudes   = 20  # Using only lowest altitude for simplicity
leadtime      = 30
lightning_type = "total" # "total" or "cloud_to_ground" or "intracloud"
n_timesteps = 3  # Number of timesteps for convLSTM

# Step 1: Split data into training days, validation days and test days
print("\nSplitting data into training and test sets...")
train_radar, train_lightning, validation_radar, validation_lightning, test_radar, test_lightning = ml_utils.get_file_splits(radar_dir, lightning_dir)

# Step 2: Load h5 file and return tensors
print("\nLoading data into tensors...")
X_train, y_train = ml_utils.load_data_to_tensors_temporal(train_radar, train_lightning, radar_dir, 
                                                 lightning_dir, n_altitudes, parameters, 
                                                 leadtime, lightning_type, n_timesteps)
X_val,   y_val   = ml_utils.load_data_to_tensors_temporal(validation_radar, validation_lightning, 
                                                 radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type, n_timesteps)
X_test , y_test  = ml_utils.load_data_to_tensors_temporal(test_radar,  test_lightning,  radar_dir, 
                                                 lightning_dir, n_altitudes, parameters, 
                                                 leadtime, lightning_type, n_timesteps)
# shape(X) = [n_samples, n_timesteps, n_lat, n_lon, n_altitudes, n_parameters]
sys.exit("asdf")



# Convert targets to binary
y_train_binary = (y_train>0).astype(float)
y_val_binary   = (y_val>0).astype(float)
y_test_binary  = (y_test>0).astype(float)

print(f"Lightning fraction in train: {np.mean(y_train_binary)*100:.2f}%")
print(f"Lightning fraction in val: {np.mean(y_val_binary)*100:.2f}%")
print(f"Lightning fraction in test: {np.mean(y_test_binary)*100:.2f}%")

# Normalizing data: Fill nan's, produce mask
print("\nNormalizing the data...")
means, stds = ml_utils.compute_normalization_parameters(X_train)
X_train_normalized, _ = ml_utils.normalize_and_preprocess(X_train, means, stds)
X_val_normalized,   _ = ml_utils.normalize_and_preprocess(X_val,   means, stds)
X_test_normalized,  _ = ml_utils.normalize_and_preprocess(X_test,  means, stds)

print("\nBefore normalization: X_train mean and std per parameter:")
print("mean: ", np.nanmean(X_train, axis=(0,1,2,3)))  
print("std: ",  np.nanstd(X_train,  axis=(0,1,2,3)))  

print("\nAfter normalization: X_train mean and std per parameter:")
print(np.mean(X_train_normalized, axis=(0,1,2,3)))  
print(np.std(X_train_normalized,  axis=(0,1,2,3)))  


# Step 4: Deal with class imbalance
neg_count = np.sum(y_train_binary == 0)
pos_count = np.sum(y_train_binary == 1)
initial_bias = np.log(pos_count / neg_count)
print(f"\nInitial class imbalance: {neg_count} negatives, {pos_count} positives")
print(f"Initial bias for loss function: {initial_bias:.4f}\n")


# Step 5: Create a simple CNN model
model = ml_utils.create_lightning_cnn(np.shape(X_test_normalized)[1:], initial_bias)
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

print("\n-- FINISHED --")

