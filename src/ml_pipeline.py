"""
ML Pipeline
-----------
Main training and evaluation script for the radar-based lightning nowcasting
pipeline.

This script:
1. Splits data into train/validation/test sets (optionally with holdout year).
2. Loads radar features and lightning targets into tensors.
3. Normalizes inputs and handles class imbalance.
4. Builds and trains the selected model architecture. 
5. Evaluates performance across multiple random seeds.
6. Optionally evaluate the model on a holdout year.

Results are aggregated and written to a summary CSV file.
"""

import numpy as np
import tensorflow as tf
import os
import random
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
import ml_utils

# Set seed number for reproducibility
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

# ===========================
#       Configuration       |
# ===========================

run_dir = "../data/processed_data/run_9_leadtime0/"
radar_dir     = run_dir + "radar/"
lightning_dir = run_dir + "lightning/"
parameters    = ['dBZ_max']  # options: 'dBZ_max', 'dBZ_mean', 'ZDR', 'KDP', 'RhoHV'
n_altitudes   = 20  
leadtime      = 0
lightning_type = "total"     # options: "total", "cloud_to_ground", "intracloud"
n_timesteps = 6              # Number of timesteps for convLSTM
model_type    = "convlstm2d" # options: "convlstm2d", "convlstm3d", "cnn2d"

# ===========================
#      Experiment Loop      |
# ===========================

csi_val        = []
threshold_val  = []
csi_test       = []
precision_test = []
recall_test    = []
accuracy_test  = []
auc_pr_test    = []
auc_roc_test   = []
for seed in seeds:
    set_seed(seed)

    # ===================
    # 1. Data Splitting |
    # ===================
    print("\nSplitting data into training and test sets...")
    train_radar, train_lightning, validation_radar, validation_lightning, test_radar, test_lightning, radar_holdout, lightning_holdout = ml_utils.get_file_splits_shuffled_by_day(radar_dir, lightning_dir, train_ratio=0.7, holdout_year=2025)

    # ===================
    # 2. Load Tensors   |
    # ===================
    print("\nLoading data into tensors...")
    # shape(X) = [n_samples, n_timesteps, n_lat, n_lon, n_altitudes, n_parameters]
    X_train, y_train = ml_utils.load_data_to_tensors_temporal(train_radar, train_lightning, radar_dir, 
                                                    lightning_dir, n_altitudes, parameters, 
                                                    leadtime, lightning_type, n_timesteps)
    X_val,   y_val   = ml_utils.load_data_to_tensors_temporal(validation_radar, validation_lightning, 
                                                    radar_dir, lightning_dir, n_altitudes, parameters, leadtime, lightning_type, n_timesteps)
    X_test , y_test  = ml_utils.load_data_to_tensors_temporal(test_radar,  test_lightning,  radar_dir, 
                                                    lightning_dir, n_altitudes, parameters, 
                                                    leadtime, lightning_type, n_timesteps)

    # Convert targets to binary
    y_train_binary = (y_train>0).astype(float)
    y_val_binary   = (y_val>0).astype(float)
    y_test_binary  = (y_test>0).astype(float)

    print(f"Lightning fraction in train: {np.mean(y_train_binary)*100:.2f}%")
    print(f"Lightning fraction in val:   {np.mean(y_val_binary)*100:.2f}%")
    print(f"Lightning fraction in test:  {np.mean(y_test_binary)*100:.2f}%")


    # ===================
    # 3. Normalization  |
    # ===================
    print("\nNormalizing the data...")
    means, stdevs = ml_utils.compute_normalization_parameters(X_train, len(parameters))
    X_train_normalized, _ = ml_utils.normalize_and_preprocess(X_train, means, stdevs, len(parameters))
    X_val_normalized,   _ = ml_utils.normalize_and_preprocess(X_val,   means, stdevs, len(parameters))
    X_test_normalized,  _ = ml_utils.normalize_and_preprocess(X_test,  means, stdevs, len(parameters))


    # ====================
    # 4. Class Imbalance |
    # ====================
    neg_count = np.sum(y_train_binary == 0)
    pos_count = np.sum(y_train_binary == 1)
    initial_bias = np.log(pos_count / neg_count)
    print(f"\nInitial class imbalance: {neg_count} negatives, {pos_count} positives")
    print(f"Initial bias for loss function: {initial_bias:.4f}\n")


    # ===================
    # 5. Model Setup    |
    # ===================

    if model_type=="convlstm2d":
        model = ml_utils.create_lightning_convLSTM2D(np.shape(X_test_normalized)[1:], initial_bias)
    elif model_type=="convlstm3d":
        transpose_indices = (0, 1, 4, 2, 3, 5)
        X_train_normalized = np.transpose(X_train_normalized, transpose_indices)
        X_val_normalized   = np.transpose(X_val_normalized,   transpose_indices)
        X_test_normalized  = np.transpose(X_test_normalized,  transpose_indices)
        model = ml_utils.create_lightning_convLSTM3D(np.shape(X_test_normalized)[1:], initial_bias)
    elif model_type=="cnn2d":
        X_train_normalized = X_train_normalized[:, -1, ...]
        X_val_normalized   = X_val_normalized[:, -1, ...]
        X_test_normalized  = X_test_normalized[:, -1, ...]
        model = ml_utils.create_lightning_cnn((np.shape(X_test_normalized)[1:]), initial_bias)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['precision', 'recall']
    )

    # ===================
    # 6. Training       |
    # ===================

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

    # ===================
    # 7. Evaluation     |
    # ===================
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("\n" + "="*50)

    # Get predictions
    y_test_pred = model.predict(X_test_normalized)
    y_val_pred  = model.predict(X_val_normalized)

    # Select decision threshold by maximizing CSI on the validation set
    threshold_csi = ml_utils.evaluate_threshold(y_val_binary, y_val_pred)

    # Find the threshold with highest CSI
    best_threshold, best_csi = max(threshold_csi, key=lambda x: x[1])
    threshold_val.append(best_threshold)
    csi_val.append(best_csi)
    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"Best validation CSI: {best_csi:.4f}")

    # Make final binary predictions
    y_pred_binary = (y_test_pred > best_threshold).astype(int)

    # Generate detailed output
    csi, precision, recall, accuracy = ml_utils.print_detailed_results(y_test_binary, y_pred_binary)
    csi_test.append(csi)
    precision_test.append(precision)
    recall_test.append(recall)
    accuracy_test.append(accuracy)

    # Calculate threshold-independent metrics
    auc_pr  = average_precision_score(y_test_binary.flatten(), y_test_pred.flatten())
    auc_roc = roc_auc_score(y_test_binary.flatten(), y_test_pred.flatten())
    auc_pr_test.append(auc_pr)
    auc_roc_test.append(auc_roc)

    print(f"\nAUC-PR: {auc_pr:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # Plot X_true, y_true and y_pred
    print("\n-- PLOTTING RESULTS --")
    ml_utils.visualize_results(X_test, y_test_binary, y_pred_binary, run_dir)

    # Plot number of lightnings as function of time
    ml_utils.visualize_timeline(y_test_binary, y_pred_binary, run_dir)

# ==================================
# 8. Aggregate results across seeds|
# ==================================
def mean_std(x):
    return float(np.mean(x)), float(np.std(x))

csi_val_mean,   csi_val_std   = mean_std(csi_val)
csi_test_mean,  csi_test_std  = mean_std(csi_test)
precision_mean, precision_std = mean_std(precision_test)
recall_mean,    recall_std    = mean_std(recall_test)
accuracy_mean,  accuracy_std  = mean_std(accuracy_test)
aupr_mean,      aupr_std      = mean_std(auc_pr_test)
auroc_mean,     auroc_std     = mean_std(auc_roc_test)
threshold_mean, threshold_std = mean_std(threshold_val)

summary_row = {
    "model": model_type,
    "parameters": ",".join(parameters),
    "n_timesteps": n_timesteps,
    "n_altitudes": n_altitudes,
    "leadtime_min": leadtime,
    "lightning_type": lightning_type,
    "n_seeds": len(seeds),

    # aggregated metrics (mean ± std)
    "CSI_val_mean":    np.round(csi_val_mean, 3),   "CSI_val_std":    np.round(csi_val_std, 3),
    "CSI_test_mean":   np.round(csi_test_mean, 3),  "CSI_test_std":   np.round(csi_test_std, 3),
    "Precision_mean":  np.round(precision_mean, 3), "Precision_std":  np.round(precision_std, 3),
    "Recall_mean":     np.round(recall_mean, 3),    "Recall_std":     np.round(recall_std, 3),
    "Accuracy_mean":   np.round(accuracy_mean, 3),  "Accuracy_std":   np.round(accuracy_std, 3),
    "AUC_PR_mean":     np.round(aupr_mean, 3),      "AUC_PR_std":     np.round(aupr_std, 3),
    "AUC_ROC_mean":    np.round(auroc_mean, 3),     "AUC_ROC_std":    np.round(auroc_std, 3),
    "Thresh_val_mean": np.round(threshold_mean, 3), "Thresh_val_std": np.round(threshold_std, 3),
}

# ====================================
# 9. Write aggregated results to CSV |
# ====================================
results_file = run_dir + "results_summary.csv"
if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
else:
    df = pd.DataFrame([summary_row])
df.to_csv(results_file, index=False)

print("\n=== SUMMARY over seeds ===")
print(f"CSI_test: {csi_test_mean:.4f} ± {csi_test_std:.4f} | "
      f"Prec: {precision_mean:.3f} ± {precision_std:.3f} | "
      f"Rec: {recall_mean:.3f} ± {recall_std:.3f} | "
      f"AUC-PR: {aupr_mean:.3f} ± {aupr_std:.3f} | "
      f"AUC-ROC: {auroc_mean:.3f} ± {auroc_std:.3f}")

# =================================
# 10. Holdout year evaluation     |
# =================================

# Note:
# Holdout evaluation is performed using the final trained model
# from the last seed iteration. The oldout visualization therefore 
# represents one trained instance, not the mean performance across
# seeds.

ml_utils.evaluate_holdout_year(model, radar_holdout, lightning_holdout,
                              radar_dir, lightning_dir, n_altitudes,
                              parameters, leadtime, lightning_type,
                              n_timesteps, means, stdevs, best_threshold,
                              model_type, run_dir)

print("\n-- FINISHED --")

