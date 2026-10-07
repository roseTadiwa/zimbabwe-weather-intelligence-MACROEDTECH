
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
# 6. Load model
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
# 8. Load target scaler
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
        "city"
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
# 11. Add temporal variables
# ============================================================

analysis_df["month"] = (
    analysis_df["time"].dt.month
)

analysis_df["month_name"] = (
    analysis_df["time"].dt.month_name()
)

analysis_df["hour"] = (
    analysis_df["time"].dt.hour
)


# ============================================================
# 12. Monthly error analysis
# ============================================================

print(
    "\nCreating monthly error analysis..."
)

monthly_results = (
    analysis_df
    .groupby(
        ["month", "month_name"],
        sort=True
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
        )
    )
    .reset_index()
)


monthly_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lstm_monthly_error_results.csv"
    ),
    index=False
)


# ============================================================
# 13. Hourly error analysis
# ============================================================

print(
    "Creating hourly error analysis..."
)

hourly_results = (
    analysis_df
    .groupby(
        "hour",
        sort=True
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
        )
    )
    .reset_index()
)


hourly_results.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "lstm_hourly_error_results.csv"
    ),
    index=False
)


# ============================================================
# 14. Monthly MAE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    monthly_results["month_name"],
    monthly_results["MAE_C"]
)

plt.xlabel("Month")
plt.ylabel("MAE (°C)")

plt.title(
    "LSTM Monthly Prediction Error (2025 Test Set)"
)

plt.xticks(
    rotation=45
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_monthly_mae.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 15. Hourly MAE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    hourly_results["hour"],
    hourly_results["MAE_C"]
)

plt.xlabel("Hour of Day")
plt.ylabel("MAE (°C)")

plt.title(
    "LSTM Hourly Prediction Error (2025 Test Set)"
)

plt.xticks(
    range(24)
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_hourly_mae.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 16. Monthly bias
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    monthly_results["month_name"],
    monthly_results["Mean_Error_C"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Month")
plt.ylabel("Mean Error (°C)")

plt.title(
    "LSTM Monthly Prediction Bias (2025 Test Set)"
)

plt.xticks(
    rotation=45
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_monthly_bias.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 17. Hourly bias
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    hourly_results["hour"],
    hourly_results["Mean_Error_C"]
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Hour of Day")
plt.ylabel("Mean Error (°C)")

plt.title(
    "LSTM Hourly Prediction Bias (2025 Test Set)"
)

plt.xticks(
    range(24)
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lstm_hourly_bias.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 18. Print results
# ============================================================

print("\n")
print("=" * 75)
print("LSTM MONTHLY ERROR ANALYSIS")
print("=" * 75)

print(
    monthly_results.to_string(
        index=False
    )
)


print("\n")
print("=" * 75)
print("LSTM HOURLY ERROR ANALYSIS")
print("=" * 75)

print(
    hourly_results.to_string(
        index=False
    )
)


# ============================================================
# 19. Files saved
# ============================================================

print("\n")
print("=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    "reports/lstm_monthly_error_results.csv"
)

print(
    "reports/lstm_hourly_error_results.csv"
)

print(
    "reports/figures/lstm_monthly_mae.png"
)

print(
    "reports/figures/lstm_hourly_mae.png"
)

print(
    "reports/figures/lstm_monthly_bias.png"
)

print(
    "reports/figures/lstm_hourly_bias.png"
)

print("\nTemporal error analysis completed successfully.")

