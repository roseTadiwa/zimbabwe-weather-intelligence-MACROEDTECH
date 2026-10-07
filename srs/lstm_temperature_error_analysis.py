
import os
import numpy as np
import pandas as pd
import keras
import joblib
import matplotlib.pyplot as plt


# ============================================================
# 1. File paths
# ============================================================

MODEL_PATH = "models/lstm_weather_model.keras"
DATA_PATH = "data/processed/model_data/test.csv"
X_TEST_PATH = "data/processed/lstm_data/X_test.npy"
Y_TEST_PATH = "data/processed/lstm_data/y_test.npy"
TARGET_SCALER_PATH = "data/processed/lstm_data/target_scaler.pkl"

OUTPUT_DIR = "reports"
FIGURES_DIR = "reports/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# 2. Load test data
# ============================================================

print("Loading test data...")

test_df = pd.read_csv(DATA_PATH)

test_df["time"] = pd.to_datetime(
    test_df["time"]
)

test_df = test_df.sort_values(
    ["city", "time"]
).reset_index(drop=True)


# ============================================================
# 3. Load LSTM test sequences and targets
# ============================================================

print("Loading LSTM test sequences...")

X_test = np.load(
    X_TEST_PATH
)

y_test_scaled = np.load(
    Y_TEST_PATH
)


# ============================================================
# 4. Reconstruct sequence-aligned observations
# ============================================================

LOOKBACK = 24

aligned_rows = []

for city, city_df in test_df.groupby("city"):

    city_df = city_df.sort_values(
        "time"
    ).reset_index(drop=True)

    aligned_city_df = city_df.iloc[
        LOOKBACK:
    ].copy()

    aligned_rows.append(
        aligned_city_df
    )


aligned_df = pd.concat(
    aligned_rows,
    ignore_index=True
)


# ============================================================
# 5. Verify alignment
# ============================================================

print("\nAlignment check:")

print(
    "Expected observations:",
    len(y_test_scaled)
)

print(
    "Aligned observations :",
    len(aligned_df)
)

if len(aligned_df) != len(y_test_scaled):

    raise ValueError(
        "The number of aligned observations "
        "does not match the LSTM test targets."
    )


# ============================================================
# 6. Load trained LSTM model
# ============================================================

print("\nLoading trained LSTM model...")

model = keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# 7. Generate predictions
# ============================================================

print("Generating predictions...")

pred_scaled = model.predict(
    X_test,
    verbose=1
).flatten()


# ============================================================
# 8. Load target scaler using JOBLIB
# ============================================================

print("\nLoading target scaler...")

target_scaler = joblib.load(
    TARGET_SCALER_PATH
)


# ============================================================
# 9. Convert values back to Celsius
# ============================================================

actual_temperature = (
    target_scaler.inverse_transform(
        y_test_scaled.reshape(-1, 1)
    ).flatten()
)

predicted_temperature = (
    target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).flatten()
)


# ============================================================
# 10. Build analysis dataframe
# ============================================================

analysis_df = aligned_df[
    [
        "time",
        "city",
        "temperature_2m",
        "target_temperature_1h"
    ]
].copy()

analysis_df[
    "actual_temperature"
] = actual_temperature

analysis_df[
    "predicted_temperature"
] = predicted_temperature

analysis_df["error"] = (
    analysis_df["actual_temperature"]
    - analysis_df["predicted_temperature"]
)

analysis_df["absolute_error"] = (
    np.abs(
        analysis_df["error"]
    )
)

analysis_df["squared_error"] = (
    analysis_df["error"] ** 2
)


# ============================================================
# 11. Calculate actual one-hour temperature change
# ============================================================

analysis_df["temperature_change"] = (
    analysis_df["target_temperature_1h"]
    - analysis_df["temperature_2m"]
)

analysis_df[
    "absolute_temperature_change"
] = np.abs(
    analysis_df["temperature_change"]
)


# ============================================================
# 12. Temperature-bin analysis
# ============================================================

print(
    "\nCreating temperature-bin analysis..."
)

bins = [
    -np.inf,
    10,
    15,
    20,
    25,
    30,
    np.inf
]

labels = [
    "<10°C",
    "10–15°C",
    "15–20°C",
    "20–25°C",
    "25–30°C",
    "≥30°C"
]

analysis_df[
    "temperature_bin"
] = pd.cut(
    analysis_df["actual_temperature"],
    bins=bins,
    labels=labels,
    right=False
)


temperature_results = (
    analysis_df
    .groupby(
        "temperature_bin",
        observed=False
    )
    .agg(
        Observations=("error", "size"),
        MAE_C=(
            "absolute_error",
            "mean"
        ),
        RMSE_C=(
            "squared_error",
            lambda x: np.sqrt(
                np.mean(x)
            )
        ),
        Mean_Error_C=(
            "error",
            "mean"
        ),
        Mean_Actual_C=(
            "actual_temperature",
            "mean"
        ),
        Mean_Predicted_C=(
            "predicted_temperature",
            "mean"
        )
    )
    .reset_index()
)


temperature_results[
    "Bias_Direction"
] = np.where(
    temperature_results[
        "Mean_Error_C"
    ] > 0,
    "Underprediction",
    "Overprediction"
)


temperature_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lstm_temperature_bin_results.csv"
    ),
    index=False
)


# ============================================================
# 13. Temperature quantile analysis
# ============================================================

print(
    "Creating temperature-quantile analysis..."
)

analysis_df[
    "temperature_quantile"
] = pd.qcut(
    analysis_df["actual_temperature"],
    q=5,
    labels=[
        "Q1 - Lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5 - Highest"
    ],
    duplicates="drop"
)


quantile_results = (
    analysis_df
    .groupby(
        "temperature_quantile",
        observed=False
    )
    .agg(
        Observations=("error", "size"),
        Min_Actual_C=(
            "actual_temperature",
            "min"
        ),
        Max_Actual_C=(
            "actual_temperature",
            "max"
        ),
        Mean_Actual_C=(
            "actual_temperature",
            "mean"
        ),
        MAE_C=(
            "absolute_error",
            "mean"
        ),
        RMSE_C=(
            "squared_error",
            lambda x: np.sqrt(
                np.mean(x)
            )
        ),
        Mean_Error_C=(
            "error",
            "mean"
        )
    )
    .reset_index()
)


quantile_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lstm_temperature_quantile_results.csv"
    ),
    index=False
)


# ============================================================
# 14. Temperature-change analysis
# ============================================================

print(
    "Creating temperature-change analysis..."
)

change_bins = [
    -np.inf,
    -2,
    -1,
    -0.5,
    0.5,
    1,
    2,
    np.inf
]

change_labels = [
    "< -2°C",
    "-2 to -1°C",
    "-1 to -0.5°C",
    "-0.5 to 0.5°C",
    "0.5 to 1°C",
    "1 to 2°C",
    "> 2°C"
]

analysis_df[
    "temperature_change_bin"
] = pd.cut(
    analysis_df["temperature_change"],
    bins=change_bins,
    labels=change_labels,
    right=False
)


change_results = (
    analysis_df
    .groupby(
        "temperature_change_bin",
        observed=False
    )
    .agg(
        Observations=("error", "size"),
        MAE_C=(
            "absolute_error",
            "mean"
        ),
        RMSE_C=(
            "squared_error",
            lambda x: np.sqrt(
                np.mean(x)
            )
        ),
        Mean_Error_C=(
            "error",
            "mean"
        ),
        Mean_Temperature_Change_C=(
            "temperature_change",
            "mean"
        )
    )
    .reset_index()
)


change_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lstm_temperature_change_results.csv"
    ),
    index=False
)


# ============================================================
# 15. Save detailed analysis dataset
# ============================================================

analysis_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lstm_temperature_error_analysis.csv"
    ),
    index=False
)


# ============================================================
# 16. Figure 1: Absolute error vs actual temperature
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    analysis_df["actual_temperature"],
    analysis_df["absolute_error"],
    alpha=0.15,
    s=8
)

plt.xlabel(
    "Actual Temperature (°C)"
)

plt.ylabel(
    "Absolute Error (°C)"
)

plt.title(
    "LSTM Absolute Prediction Error vs Actual Temperature"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_error_vs_actual_temperature.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 17. Figure 2: Signed error vs actual temperature
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.scatter(
    analysis_df["actual_temperature"],
    analysis_df["error"],
    alpha=0.15,
    s=8
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Actual Temperature (°C)"
)

plt.ylabel(
    "Prediction Error (°C)"
)

plt.title(
    "LSTM Prediction Error vs Actual Temperature"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_signed_error_vs_actual_temperature.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 18. Figure 3: MAE by temperature bin
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    temperature_results[
        "temperature_bin"
    ].astype(str),
    temperature_results[
        "MAE_C"
    ]
)

plt.xlabel(
    "Actual Temperature Range"
)

plt.ylabel(
    "MAE (°C)"
)

plt.title(
    "LSTM MAE by Actual Temperature Range"
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_mae_by_temperature_bin.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 19. Figure 4: Bias by temperature bin
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    temperature_results[
        "temperature_bin"
    ].astype(str),
    temperature_results[
        "Mean_Error_C"
    ]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Actual Temperature Range"
)

plt.ylabel(
    "Mean Error (°C)"
)

plt.title(
    "LSTM Prediction Bias by Actual Temperature Range"
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_bias_by_temperature_bin.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 20. Figure 5: MAE by temperature change
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    change_results[
        "temperature_change_bin"
    ].astype(str),
    change_results[
        "MAE_C"
    ]
)

plt.xlabel(
    "Actual Temperature Change: Current Hour → Next Hour"
)

plt.ylabel(
    "MAE (°C)"
)

plt.title(
    "LSTM MAE by One-Hour Temperature Change"
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_mae_by_temperature_change.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 21. Print temperature-bin results
# ============================================================

print("\n")
print("=" * 75)
print(
    "LSTM ERROR ANALYSIS BY ACTUAL TEMPERATURE"
)
print("=" * 75)

print(
    temperature_results.to_string(
        index=False
    )
)


# ============================================================
# 22. Print quantile results
# ============================================================

print("\n")
print("=" * 75)
print(
    "LSTM ERROR ANALYSIS BY TEMPERATURE QUANTILE"
)
print("=" * 75)

print(
    quantile_results.to_string(
        index=False
    )
)


# ============================================================
# 23. Print temperature-change results
# ============================================================

print("\n")
print("=" * 75)
print(
    "LSTM ERROR ANALYSIS BY TEMPERATURE CHANGE"
)
print("=" * 75)

print(
    change_results.to_string(
        index=False
    )
)


# ============================================================
# 24. Files saved
# ============================================================

print("\n")
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    "reports/lstm_temperature_bin_results.csv"
)

print(
    "reports/lstm_temperature_quantile_results.csv"
)

print(
    "reports/lstm_temperature_change_results.csv"
)

print(
    "reports/lstm_temperature_error_analysis.csv"
)

print(
    "reports/figures/lstm_error_vs_actual_temperature.png"
)

print(
    "reports/figures/lstm_signed_error_vs_actual_temperature.png"
)

print(
    "reports/figures/lstm_mae_by_temperature_bin.png"
)

print(
    "reports/figures/lstm_bias_by_temperature_bin.png"
)

print(
    "reports/figures/lstm_mae_by_temperature_change.png"
)

print(
    "\nTemperature error analysis completed successfully."
)

