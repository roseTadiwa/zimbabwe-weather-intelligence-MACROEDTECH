
import os
import numpy as np
import pandas as pd
import keras
import joblib


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "models/lstm_weather_model.keras"

X_VAL_PATH = "data/processed/lstm_data/X_val.npy"
Y_VAL_PATH = "data/processed/lstm_data/y_val.npy"

X_TEST_PATH = "data/processed/lstm_data/X_test.npy"
Y_TEST_PATH = "data/processed/lstm_data/y_test.npy"

TARGET_SCALER_PATH = "data/processed/lstm_data/target_scaler.pkl"

RESULTS_PATH = "reports/lstm_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading validation data...")

X_val = np.load(X_VAL_PATH)
y_val = np.load(Y_VAL_PATH)

print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")


print("\nLoading test data...")

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
# FUNCTION TO CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true_scaled, y_pred_scaled):

    # Convert back to original temperature scale
    y_true = target_scaler.inverse_transform(
        y_true_scaled.reshape(-1, 1)
    ).reshape(-1)

    y_pred = target_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)
    ).reshape(-1)

    errors = y_true - y_pred

    # MAE
    mae = np.mean(
        np.abs(errors)
    )

    # RMSE
    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    # R²
    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (y_true - np.mean(y_true)) ** 2
    )

    r2 = 1 - (
        ss_res / ss_tot
    )

    return mae, rmse, r2


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

y_val_pred_scaled = model.predict(
    X_val,
    batch_size=256,
    verbose=1
)

y_val_pred_scaled = y_val_pred_scaled.reshape(-1)

print("Validation predictions generated.")


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

y_test_pred_scaled = model.predict(
    X_test,
    batch_size=256,
    verbose=1
)

y_test_pred_scaled = y_test_pred_scaled.reshape(-1)

print("Test predictions generated.")


# ============================================================
# CALCULATE VALIDATION METRICS
# ============================================================

val_mae, val_rmse, val_r2 = calculate_metrics(
    y_val,
    y_val_pred_scaled
)


# ============================================================
# CALCULATE TEST METRICS
# ============================================================

test_mae, test_rmse, test_r2 = calculate_metrics(
    y_test,
    y_test_pred_scaled
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LSTM VALIDATION AND TEST RESULTS")
print("=" * 60)

print("\nVALIDATION SET")
print("-" * 60)
print(f"MAE : {val_mae:.4f} °C")
print(f"RMSE: {val_rmse:.4f} °C")
print(f"R²  : {val_r2:.4f}")

print("\nTEST SET")
print("-" * 60)
print(f"MAE : {test_mae:.4f} °C")
print(f"RMSE: {test_rmse:.4f} °C")
print(f"R²  : {test_r2:.4f}")

print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Dataset": [
        "Validation",
        "Test"
    ],
    "Model": [
        "LSTM",
        "LSTM"
    ],
    "MAE_C": [
        val_mae,
        test_mae
    ],
    "RMSE_C": [
        val_rmse,
        test_rmse
    ],
    "R2": [
        val_r2,
        test_r2
    ]
})

os.makedirs(
    os.path.dirname(RESULTS_PATH),
    exist_ok=True
)

results.to_csv(
    RESULTS_PATH,
    index=False
)

print(
    f"\nResults saved to: {RESULTS_PATH}"
)

