import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import importlib

import ml_utils
importlib.reload(ml_utils)

# Configurations
run_dir = "../data/processed_data/run_1/"
radar_dir     = run_dir + "radar/"
lightning_dir = run_dir + "lightning/"
parameters    = ['dBZ', 'ZDR', 'KDP', 'RhoHV']
n_altitudes   = 1  # Using only lowest altitude for simplicity
leadtime      = 30
lightning_type = "cloud_to_ground" # "total" or "cloud_to_ground" or "intracloud"

# Step 1: Split data into training days and validation days
print("\nSplitting data into training and test sets...")
train_radar, train_lightning, validation_radar, validation_lightning, test_radar, test_lightning = ml_utils.get_file_splits(radar_dir, lightning_dir)

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
X_test_normalized  = normalizer(X_test)
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
print(f"Lightning weight for loss function: {lightning_weight:.2f}")

loss_function = ml_utils.create_weighted_loss(lightning_weight)

# Step 5: Create a simple CNN model
model = ml_utils.create_lightning_cnn()
model.compile(
    optimizer='adam',
    loss=loss_function,
)

# Step 6: Train the model
model.fit(
    X_train_normalized,
    y_train_log,
    epochs=10,
    batch_size=32,
    validation_data=(X_test_normalized, y_test_log),
)

# Step 7: Evaluate the model
loss = model.evaluate(X_test_normalized, y_test_log)
preds = model.predict(X_test_normalized)
pred_counts = tf.math.expm1(preds)

# Step 8: Visualize
pred_counts = tf.where(pred_counts < 0.5, tf.zeros_like(pred_counts), pred_counts)
predicted_counts = []
true_counts = []

for i in range(pred_counts.shape[0]):
        predicted_counts.append(sum(sum(pred_counts[i,:,:])))
        true_counts.append(sum(sum(y_test[i,:,:])))


plt.figure(figsize=(10, 5))
plt.plot(predicted_counts, label='Predicted Lightning Counts', color='blue')
plt.plot(true_counts, label='True Lightning Counts', color='orange')
plt.xlabel('Sample Index')
plt.ylabel('Lightning Counts')
plt.title('Predicted vs True Lightning Counts')
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# For integer counts, create bins at 0.5, 1.5, 2.5, etc.
# This way counts of 1 go in bin [0.5, 1.5), counts of 2 go in [1.5, 2.5), etc.
max_count = int(np.max(y_train))
bin_edges = np.arange(-0.5, max_count + 1.5, 1)

plt.figure(figsize=(10, 6))
plt.hist(y_train.flatten(), bins=bin_edges, alpha=0.7, edgecolor='black')
plt.xlabel('Cloud-to-Ground Lightning Count per Grid Cell')
plt.ylabel('Frequency')
plt.title('Distribution of Cloud-to-Ground Lightning Counts')
plt.xticks(range(min(max_count + 1, 20)))  # Show up to 20 on x-axis
plt.grid(True, alpha=0.3)
plt.show()

# Statistics
total_cells = y_train.size
zero_cells = np.sum(y_train == 0)
nonzero_cells = total_cells - zero_cells

print(f"Total grid cells: {total_cells}")
print(f"Cells with 0 CG lightning: {zero_cells} ({100*zero_cells/total_cells:.1f}%)")
print(f"Cells with CG lightning: {nonzero_cells} ({100*nonzero_cells/total_cells:.1f}%)")
print(f"Max CG lightning count: {np.max(y_train)}")
print(f"Mean CG count (non-zero cells): {np.mean(y_train[y_train > 0]):.2f}")