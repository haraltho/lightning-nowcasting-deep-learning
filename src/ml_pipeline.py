import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import importlib

import ml_utils
importlib.reload(ml_utils)

# Configurations
run_dir = "../data/processed_data/run_1/"
radar_dir = run_dir + "radar/"
lightning_dir = run_dir + "lightning/"
parameters = ['dBZ', 'ZDR', 'KDP', 'RhoHV']
n_altitudes = 1  # Using only lowest altitude for simplicity
leadtime = 30
lightning_type = "cloud_to_ground" # "total" or "cloud_to_ground" or "intracloud"

# Step 1: Split data into training days and validation days
print("\nSplitting data into training and test sets...")
train_radar, train_lightning, test_radar, test_lightning = ml_utils.get_file_splits(radar_dir, lightning_dir)

# Step 2: Load h5 file and return tensors
print("\nLoading data into tensors...")
X_train, y_train = ml_utils.load_data_to_tensors(train_radar, train_lightning, radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type)
X_test , y_test  = ml_utils.load_data_to_tensors(test_radar,  test_lightning,  radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type)

print(f"\nTraining data: X={X_train.shape}, y={y_train.shape}")
print(f"Test data: X={X_test.shape}, y={y_test.shape}")
print(f"Lightning fraction in train: {np.mean(y_train > 0)*100:.2f}%")
print(f"Lightning fraction in test: {np.mean(y_test > 0)*100:.2f}%")

# Step 3: Normalize the data
print("\nNormalizing the data...")
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)

print("\nBefore normalization: X_train mean and std per parameter:")
print("mean: ", np.mean(X_train, axis=(0,1,2,3)))  
print("std: ", np.std(X_train, axis=(0,1,2,3)))  

# Apply normalization
X_train_normalized = normalizer(X_train)
print("\nAfter normalization: X_train mean and std per parameter:")
print(np.mean(X_train_normalized, axis=(0,1,2,3)))  
print(np.std(X_train_normalized, axis=(0,1,2,3)))  

# Step 4: Deal with class imbalance
# Log-transform lightning targets to shrink the range
y_train_log = np.log1p(y_train)
y_test_log  = np.log1p(y_test)

print("\nLog-transformed lightning targets...")

n_total = np.size(y_train)
n_lightning = np.sum(y_train > 0)
n_non_lightning = n_total - n_lightning
lightning_weight = float(n_non_lightning / n_lightning)

loss_function = ml_utils.create_weighted_loss(lightning_weight)

# Step 5: Create a simple CNN model
model = ml_utils.create_lightning_cnn()
model.compile(
    optimizer='adam',
    loss=loss_function,
)

model.fit(
    X_train_normalized,
    y_train_log,
    epochs=10,
    batch_size=32,
    validation_data=(normalizer(X_test), y_test_log),
)

loss = model.evaluate(normalizer(X_test), y_test_log)
preds = model.predict(normalizer(X_test))
actual_counts = tf.math.expm1(preds)