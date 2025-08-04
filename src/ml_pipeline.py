import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import importlib
import os
import random

import ml_utils
importlib.reload(ml_utils)

# Set seed number for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)

# Configurations
run_dir = "../data/processed_data/run_3_2022_2023_2024/"
radar_dir     = run_dir + "radar/"
lightning_dir = run_dir + "lightning/"
parameters    = ['dBZ', 'ZDR', 'KDP', 'RhoHV']
n_altitudes   = 20  # Using only lowest altitude for simplicity
leadtime      = 30
lightning_type = "cloud_to_ground" # "total" or "cloud_to_ground" or "intracloud"

# Step 1: Split data into training days, validation days and test days
print("\nSplitting data into training and test sets...")
train_radar, train_lightning, validation_radar, validation_lightning, test_radar, test_lightning = ml_utils.get_file_splits(radar_dir, lightning_dir)

# Step 2: Load h5 file and return tensors
print("\nLoading data into tensors...")
X_train, y_train = ml_utils.load_data_to_tensors(train_radar, train_lightning, radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type)
X_val,   y_val   = ml_utils.load_data_to_tensors(validation_radar, validation_lightning, radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type)
X_test , y_test  = ml_utils.load_data_to_tensors(test_radar,  test_lightning,  radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type)

# Convert targets to binary
y_train_binary = (y_train>0).astype(float)
y_val_binary   = (y_val>0).astype(float)
y_test_binary  = (y_test>0).astype(float)

print(f"Lightning fraction in train: {np.mean(y_train_binary)*100:.2f}%")
print(f"Lightning fraction in val: {np.mean(y_val_binary)*100:.2f}%")
print(f"Lightning fraction in test: {np.mean(y_test_binary)*100:.2f}%")

# Step 3: Normalize the data
print("\nNormalizing the data...")
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)

print("\nBefore normalization: X_train mean and std per parameter:")
print("mean: ", np.mean(X_train, axis=(0,1,2,3)))  
print("std: ", np.std(X_train, axis=(0,1,2,3)))  

# Apply normalization
X_train_normalized = normalizer(X_train)
X_val_normalized   = normalizer(X_val)
X_test_normalized  = normalizer(X_test)
print("\nAfter normalization: X_train mean and std per parameter:")
print(np.mean(X_train_normalized, axis=(0,1,2,3)))  
print(np.std(X_train_normalized, axis=(0,1,2,3)))  

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

# Plot X_true, y_true and y_pred
print("\n-- PLOTTING RESULTS --")
# ml_utils.visualize_results(X_test, y_test_binary, y_pred_binary, run_dir)

print("\n-- FINISHED --")

