import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed/zimbabwe_weather_model.csv"
RESULTS_PATH = "reports/persistence_results.csv"

VAL_END = "2024-12-31 23:00:00"
LOOKBACK = 24


# ============================================================
# LOAD DATA
# ============================================================

print("Loading model dataset...")

df = pd.read_csv(DATA_PATH)

df["time"] = pd.to_datetime(df["time"])

df = df.sort_values(
    ["city", "time"]
).reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Cities: {sorted(df['city'].unique())}")


# ============================================================
# SELECT 2025 TEST DATA
# ============================================================

test_df = df[
    df["time"] > VAL_END
].copy()

print("\nTest dataset:")
print(f"Shape: {test_df.shape}")
print(
    f"Date range: "
    f"{test_df['time'].min()} to {test_df['time'].max()}"
)


# ============================================================
# ALIGN WITH LSTM TEST SEQUENCES
# ============================================================

# The LSTM uses a 24-hour lookback.
# Therefore, the first 24 test observations for each city
# are excluded so that the persistence baseline is evaluated
# on exactly the same observations as the LSTM.

aligned_test = []

for city in sorted(test_df["city"].unique()):

    city_data = test_df[
        test_df["city"] == city
    ].sort_values("time").copy()

    city_aligned = city_data.iloc[LOOKBACK:].copy()

    aligned_test.append(city_aligned)


aligned_test = pd.concat(
    aligned_test,
    ignore_index=True
)

print("\nLSTM-aligned test data:")
print(f"Shape: {aligned_test.shape}")

print("\nObservations by city:")
print(
    aligned_test["city"].value_counts().sort_index()
)


# ============================================================
# PERSISTENCE PREDICTIONS
# ============================================================

# Persistence assumption:
#
# Next-hour temperature = current-hour temperature
#
# Therefore:
#
# prediction = temperature_2m

actual = aligned_test[
    "target_temperature_1h"
].to_numpy()

predicted = aligned_test[
    "temperature_2m"
].to_numpy()


# ============================================================
# CALCULATE METRICS
# ============================================================

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


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PERSISTENCE BASELINE RESULTS")
print("=" * 60)

print(f"Test MAE : {mae:.4f} °C")
print(f"Test RMSE: {rmse:.4f} °C")
print(f"Test R²  : {r2:.4f}")

print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(RESULTS_PATH),
    exist_ok=True
)

results = pd.DataFrame({
    "Model": ["Persistence"],
    "Test_MAE": [mae],
    "Test_RMSE": [rmse],
    "Test_R2": [r2],
    "Test_Observations": [len(actual)]
})

results.to_csv(
    RESULTS_PATH,
    index=False
)

print(
    f"\nResults saved to: {RESULTS_PATH}"
)