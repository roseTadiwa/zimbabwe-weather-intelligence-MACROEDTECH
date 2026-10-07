
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import joblib


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/lstm_weather_model.keras"

X_TEST_PATH = "data/processed/lstm_data/X_test.npy"
Y_TEST_PATH = "data/processed/lstm_data/y_test.npy"

TARGET_SCALER_PATH = "data/processed/lstm_data/target_scaler.pkl"

FIGURES_PATH = "reports/figures"

RESULTS_PATH = "reports/lstm_prediction_diagnostics.csv"


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(FIGURES_PATH, exist_ok=True)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading test data...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================
# LOAD TARGET SCALER
# ============================================================

print("\nLoading target scaler...")

target_scaler = joblib.load(TARGET_SCALER_PATH)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained LSTM model...")

model = keras.models.load_model(MODEL_PATH)

print("LSTM model loaded successfully.")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred_scaled = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

y_pred_scaled = y_pred_scaled.reshape(-1)

print("Predictions generated successfully.")


# ============================================================
# CONVERT TO ORIGINAL TEMPERATURE SCALE
# ============================================================

print("\nConverting values back to °C...")

y_test_original = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(-1)

y_pred_original = target_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).reshape(-1)


# ============================================================
# CALCULATE ERRORS
# ============================================================

errors = y_test_original - y_pred_original

absolute_errors = np.abs(errors)


# ============================================================
# ERROR STATISTICS
# ============================================================

mae = np.mean(absolute_errors)

rmse = np.sqrt(
    np.mean(errors ** 2)
)

mean_error = np.mean(errors)

median_error = np.median(errors)

max_error = np.max(absolute_errors)

actual_mean = np.mean(y_test_original)

predicted_mean = np.mean(y_pred_original)

actual_std = np.std(y_test_original)

predicted_std = np.std(y_pred_original)


# ============================================================
# DISPLAY STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("LSTM PREDICTION DIAGNOSTICS")
print("=" * 60)

print(f"MAE                 : {mae:.4f} °C")
print(f"RMSE                : {rmse:.4f} °C")
print(f"Mean Error          : {mean_error:.4f} °C")
print(f"Median Error        : {median_error:.4f} °C")
print(f"Maximum Absolute Error: {max_error:.4f} °C")

print("\nACTUAL TEMPERATURE")
print("-" * 60)
print(f"Mean                : {actual_mean:.4f} °C")
print(f"Standard Deviation  : {actual_std:.4f} °C")
print(f"Minimum             : {np.min(y_test_original):.4f} °C")
print(f"Maximum             : {np.max(y_test_original):.4f} °C")

print("\nPREDICTED TEMPERATURE")
print("-" * 60)
print(f"Mean                : {predicted_mean:.4f} °C")
print(f"Standard Deviation  : {predicted_std:.4f} °C")
print(f"Minimum             : {np.min(y_pred_original):.4f} °C")
print(f"Maximum             : {np.max(y_pred_original):.4f} °C")

print("=" * 60)


# ============================================================
# 1. ACTUAL VS PREDICTED TEMPERATURE
# ============================================================

print("\nCreating actual vs predicted plot...")

sample_size = min(1000, len(y_test_original))

plt.figure(figsize=(12, 6))

plt.plot(
    y_test_original[:sample_size],
    label="Actual Temperature"
)

plt.plot(
    y_pred_original[:sample_size],
    label="LSTM Prediction"
)

plt.xlabel("Test Observation")
plt.ylabel("Temperature (°C)")
plt.title("LSTM Actual vs Predicted Temperature")
plt.legend()
plt.tight_layout()

actual_predicted_path = os.path.join(
    FIGURES_PATH,
    "lstm_actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300
)

plt.close()

print(
    f"Saved: {actual_predicted_path}"
)


# ============================================================
# 2. ERROR DISTRIBUTION
# ============================================================

print("\nCreating error distribution plot...")

plt.figure(figsize=(10, 6))

plt.hist(
    errors,
    bins=50
)

plt.xlabel("Prediction Error (°C)")
plt.ylabel("Frequency")
plt.title("LSTM Prediction Error Distribution")
plt.tight_layout()

error_distribution_path = os.path.join(
    FIGURES_PATH,
    "lstm_error_distribution.png"
)

plt.savefig(
    error_distribution_path,
    dpi=300
)

plt.close()

print(
    f"Saved: {error_distribution_path}"
)


# ============================================================
# 3. ACTUAL VS PREDICTED SCATTER PLOT
# ============================================================

print("\nCreating actual vs predicted scatter plot...")

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test_original,
    y_pred_original,
    alpha=0.3
)

min_value = min(
    np.min(y_test_original),
    np.min(y_pred_original)
)

max_value = max(
    np.max(y_test_original),
    np.max(y_pred_original)
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("LSTM Actual vs Predicted Temperature")

plt.tight_layout()

scatter_path = os.path.join(
    FIGURES_PATH,
    "lstm_actual_vs_predicted_scatter.png"
)

plt.savefig(
    scatter_path,
    dpi=300
)

plt.close()

print(
    f"Saved: {scatter_path}"
)


# ============================================================
# SAVE DIAGNOSTIC RESULTS
# ============================================================

diagnostics = pd.DataFrame({
    "Metric": [
        "MAE_C",
        "RMSE_C",
        "Mean_Error_C",
        "Median_Error_C",
        "Maximum_Absolute_Error_C",
        "Actual_Mean_C",
        "Actual_Std_C",
        "Actual_Min_C",
        "Actual_Max_C",
        "Predicted_Mean_C",
        "Predicted_Std_C",
        "Predicted_Min_C",
        "Predicted_Max_C"
    ],
    "Value": [
        mae,
        rmse,
        mean_error,
        median_error,
        max_error,
        actual_mean,
        actual_std,
        np.min(y_test_original),
        np.max(y_test_original),
        predicted_mean,
        predicted_std,
        np.min(y_pred_original),
        np.max(y_pred_original)
    ]
})

diagnostics.to_csv(
    RESULTS_PATH,
    index=False
)

print(
    f"\nDiagnostic results saved to: {RESULTS_PATH}"
)

print("\nLSTM diagnostics completed successfully.")

