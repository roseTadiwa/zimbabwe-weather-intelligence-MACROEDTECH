
import os
import numpy as np
import pandas as pd
import keras
import joblib


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/processed/zimbabwe_weather_model.csv"

MODEL_PATH = "models/lstm_weather_model.keras"

X_TEST_PATH = "data/processed/lstm_data/X_test.npy"
Y_TEST_PATH = "data/processed/lstm_data/y_test.npy"

TARGET_SCALER_PATH = "data/processed/lstm_data/target_scaler.pkl"

RESULTS_PATH = "reports/lstm_city_results.csv"


# ============================================================
# SETTINGS
# ============================================================

LOOKBACK = 24

TRAIN_END = "2022-12-31 23:00:00"
VAL_END = "2024-12-31 23:00:00"


# ============================================================
# LOAD ORIGINAL MODEL DATASET
# ============================================================

print("Loading original model dataset...")

df = pd.read_csv(DATA_PATH)

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    ["city", "time"]
).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Cities: {sorted(df['city'].unique())}")


# ============================================================
# RECREATE TEST DATA SPLIT
# ============================================================

test_df = df[
    df["time"] > pd.Timestamp(VAL_END)
].copy()

print("\nTest dataframe shape:")
print(test_df.shape)


# ============================================================
# RECREATE CITY LABELS FOR TEST SEQUENCES
# ============================================================

print("\nReconstructing city labels for test sequences...")

sequence_cities = []

for city in sorted(test_df["city"].unique()):

    city_data = test_df[
        test_df["city"] == city
    ].copy()

    city_data = city_data.sort_values("time")

    number_of_sequences = len(city_data) - LOOKBACK

    sequence_cities.extend(
        [city] * number_of_sequences
    )

sequence_cities = np.array(sequence_cities)

print(
    f"Number of reconstructed sequence labels: "
    f"{len(sequence_cities)}"
)


# ============================================================
# LOAD TEST ARRAYS
# ============================================================

print("\nLoading test sequences...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================
# VERIFY SEQUENCE LABEL COUNT
# ============================================================

if len(sequence_cities) != len(y_test):

    raise ValueError(
        "Number of reconstructed city labels does not "
        "match the number of test sequences."
    )

print("City-to-sequence mapping verified.")


# ============================================================
# LOAD TARGET SCALER
# ============================================================

print("\nLoading target scaler...")

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained LSTM model...")

model = keras.models.load_model(
    MODEL_PATH
)

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
# CONVERT BACK TO ORIGINAL TEMPERATURE SCALE
# ============================================================

print("\nConverting values back to °C...")

y_test_original = target_scaler.inverse_transform(
    y_test.reshape(-1, 1)
).reshape(-1)

y_pred_original = target_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).reshape(-1)


# ============================================================
# CALCULATE CITY-LEVEL METRICS
# ============================================================

print("\nCalculating city-level metrics...")

city_results = []

for city in sorted(
    np.unique(sequence_cities)
):

    mask = sequence_cities == city

    actual = y_test_original[mask]

    predicted = y_pred_original[mask]

    errors = actual - predicted

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    r2 = 1 - (
        ss_res / ss_tot
    )

    mean_error = np.mean(
        errors
    )

    city_results.append({
        "City": city,
        "Test_Sequences": len(actual),
        "MAE_C": mae,
        "RMSE_C": rmse,
        "R2": r2,
        "Mean_Error_C": mean_error
    })


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    city_results
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("LSTM CITY-LEVEL TEST RESULTS")
print("=" * 80)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("=" * 80)


# ============================================================
# SAVE RESULTS
# ============================================================

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

print("\nCity-level LSTM evaluation completed successfully.")

