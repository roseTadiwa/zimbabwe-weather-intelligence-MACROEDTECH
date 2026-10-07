import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "reports/lstm_v2_temperature_error_analysis.csv"
OUTPUT_PATH = "reports/lstm_v2_city_results.csv"


# ============================================================
# LOAD V2 PREDICTION-LEVEL RESULTS
# ============================================================

print("Loading LSTM v2 prediction-level results...")

df = pd.read_csv(INPUT_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Cities: {df['city'].unique().tolist()}")


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = [
    "city",
    "actual_temperature",
    "predicted_temperature",
    "error"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CITY-LEVEL EVALUATION
# ============================================================

results = []

for city, city_df in df.groupby("city"):

    actual = city_df["actual_temperature"].to_numpy()
    predicted = city_df["predicted_temperature"].to_numpy()

    error = actual - predicted

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    r2 = 1 - (ss_res / ss_tot)

    mean_error = np.mean(error)

    results.append({
        "City": city,
        "Test_Sequences": len(city_df),
        "MAE_C": mae,
        "RMSE_C": rmse,
        "R2": r2,
        "Mean_Error_C": mean_error
    })


# ============================================================
# CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "City"
).reset_index(drop=True)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nLSTM v2 CITY-LEVEL EVALUATION")
print("=" * 75)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("=" * 75)

print(
    f"\nResults saved to: {OUTPUT_PATH}"
)