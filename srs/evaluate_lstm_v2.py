import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras


# ============================================================
# PATHS
# ============================================================

LSTM_DATA_DIR = "data/processed/lstm_data_v2"

MODEL_PATH = "models/lstm_weather_model_v2.keras"

RESULTS_PATH = "reports/lstm_v2_results.csv"


# ============================================================
# LOAD TEST DATA
# ============================================================

print("Loading test data...")

X_test = np.load(
    f"{LSTM_DATA_DIR}/X_test.npy"
)

y_test = np.load(
    f"{LSTM_DATA_DIR}/y_test.npy"
)


print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading improved LSTM model...")

model = keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred_scaled = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)


y_pred_scaled = y_pred_scaled.ravel()


# ============================================================
# LOAD TARGET SCALER
# ============================================================

target_scaler = joblib.load(
    f"{LSTM_DATA_DIR}/target_scaler.pkl"
)


# ============================================================
# INVERSE TRANSFORM
# ============================================================

y_test_actual = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).ravel()


y_pred_actual = target_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).ravel()


# ============================================================
# CALCULATE METRICS
# ============================================================

errors = (
    y_test_actual
    - y_pred_actual
)

absolute_errors = np.abs(
    errors
)

squared_errors = (
    errors ** 2
)


mae = np.mean(
    absolute_errors
)

rmse = np.sqrt(
    np.mean(squared_errors)
)


ss_res = np.sum(
    squared_errors
)

ss_tot = np.sum(
    (
        y_test_actual
        - np.mean(y_test_actual)
    ) ** 2
)

r2 = 1 - (
    ss_res / ss_tot
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED LSTM TEST RESULTS")
print("=" * 70)

print(
    f"Test MAE:  {mae:.4f} °C"
)

print(
    f"Test RMSE: {rmse:.4f} °C"
)

print(
    f"Test R²:   {r2:.4f}"
)

print("=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Model": ["LSTM_v2"],
    "Test_MAE": [mae],
    "Test_RMSE": [rmse],
    "Test_R2": [r2]
})


os.makedirs(
    "reports",
    exist_ok=True
)

results.to_csv(
    RESULTS_PATH,
    index=False
)


print(
    f"\nResults saved to: {RESULTS_PATH}"
)