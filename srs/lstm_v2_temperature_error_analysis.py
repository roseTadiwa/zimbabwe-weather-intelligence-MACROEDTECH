import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

MODEL_DATA_PATH = "data/processed/zimbabwe_weather_model.csv"

LSTM_DATA_DIR = "data/processed/lstm_data_v2"

MODEL_PATH = "models/lstm_weather_model_v2.keras"

REPORTS_DIR = "reports"

FIGURES_DIR = "reports/figures"


os.makedirs(
    REPORTS_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading model dataset...")

df = pd.read_csv(
    MODEL_DATA_PATH,
    parse_dates=["time"]
)


# ============================================================
# LOAD TEST SEQUENCES
# ============================================================

print("Loading LSTM v2 test sequences...")

X_test = np.load(
    f"{LSTM_DATA_DIR}/X_test.npy"
)

y_test_scaled = np.load(
    f"{LSTM_DATA_DIR}/y_test.npy"
)


print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test_scaled.shape}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading LSTM v2 model...")

model = keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred_scaled = model.predict(
    X_test,
    batch_size=256,
    verbose=1
).ravel()


# ============================================================
# LOAD TARGET SCALER
# ============================================================

target_scaler = joblib.load(
    f"{LSTM_DATA_DIR}/target_scaler.pkl"
)


# ============================================================
# INVERSE TRANSFORM
# ============================================================

y_actual = target_scaler.inverse_transform(
    y_test_scaled.reshape(-1, 1)
).ravel()


y_predicted = target_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).ravel()


# ============================================================
# ALIGN TEST DATA
# ============================================================

test_df = df[
    df["time"] > "2024-12-31 23:00:00"
].copy()


test_df = test_df.sort_values(
    ["city", "time"]
).reset_index(drop=True)


# Remove the first 24 observations for each city
# because the LSTM sequence uses a 24-hour lookback.

aligned_parts = []

for city in sorted(test_df["city"].unique()):

    city_df = test_df[
        test_df["city"] == city
    ].sort_values("time").reset_index(drop=True)

    city_df = city_df.iloc[24:].copy()

    aligned_parts.append(city_df)


aligned_test_df = pd.concat(
    aligned_parts,
    ignore_index=True
)


# ============================================================
# CHECK ALIGNMENT
# ============================================================

print("\nAlignment check:")

print(
    f"Aligned observations: "
    f"{len(aligned_test_df)}"
)

print(
    f"Predictions: "
    f"{len(y_predicted)}"
)


if len(aligned_test_df) != len(y_predicted):

    raise ValueError(
        "Alignment error: number of observations "
        "does not match number of predictions."
    )


# ============================================================
# CREATE ANALYSIS DATAFRAME
# ============================================================

analysis_df = pd.DataFrame({
    "time": aligned_test_df["time"].values,
    "city": aligned_test_df["city"].values,
    "actual_temperature": y_actual,
    "predicted_temperature": y_predicted
})


# ============================================================
# CALCULATE ERRORS
# ============================================================

analysis_df["error"] = (
    analysis_df["actual_temperature"]
    - analysis_df["predicted_temperature"]
)

analysis_df["absolute_error"] = (
    np.abs(analysis_df["error"])
)

analysis_df["squared_error"] = (
    analysis_df["error"] ** 2
)


# ============================================================
# CALCULATE ONE-HOUR TEMPERATURE CHANGE
# ============================================================

analysis_df["temperature_change_1h"] = (
    analysis_df
    .groupby("city")["actual_temperature"]
    .diff()
)


analysis_df = analysis_df.dropna(
    subset=["temperature_change_1h"]
).reset_index(drop=True)


# ============================================================
# FUNCTION FOR METRICS
# ============================================================

def calculate_metrics(group):

    actual = group["actual_temperature"].values
    predicted = group["predicted_temperature"].values

    errors = actual - predicted

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    mean_error = np.mean(
        errors
    )

    return pd.Series({
        "Observations": len(group),
        "MAE_C": mae,
        "RMSE_C": rmse,
        "Mean_Error_C": mean_error,
        "Mean_Actual_C": np.mean(actual),
        "Mean_Predicted_C": np.mean(predicted)
    })


# ============================================================
# TEMPERATURE BINS
# ============================================================

temperature_bins = [
    -np.inf,
    10,
    15,
    20,
    25,
    30,
    np.inf
]

temperature_labels = [
    "<10°C",
    "10–15°C",
    "15–20°C",
    "20–25°C",
    "25–30°C",
    "≥30°C"
]


analysis_df["temperature_bin"] = pd.cut(
    analysis_df["actual_temperature"],
    bins=temperature_bins,
    labels=temperature_labels,
    right=False
)


temperature_results = (
    analysis_df
    .groupby(
        "temperature_bin",
        observed=False
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)


temperature_results["Bias_Direction"] = np.where(
    temperature_results["Mean_Error_C"] > 0,
    "Underprediction",
    "Overprediction"
)


# ============================================================
# SAVE TEMPERATURE BIN RESULTS
# ============================================================

temperature_results.to_csv(
    f"{REPORTS_DIR}/lstm_v2_temperature_bin_results.csv",
    index=False
)


# ============================================================
# TEMPERATURE QUANTILES
# ============================================================

analysis_df["temperature_quantile"] = pd.qcut(
    analysis_df["actual_temperature"],
    q=5,
    labels=[
        "Q1 - Lowest",
        "Q2",
        "Q3",
        "Q4",
        "Q5 - Highest"
    ]
)


quantile_results = (
    analysis_df
    .groupby(
        "temperature_quantile",
        observed=False
    )
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)


quantile_results.to_csv(
    f"{REPORTS_DIR}/lstm_v2_temperature_quantile_results.csv",
    index=False
)


# ============================================================
# TEMPERATURE CHANGE BINS
# ============================================================

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


analysis_df["temperature_change_bin"] = pd.cut(
    analysis_df["temperature_change_1h"],
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
    .apply(
        calculate_metrics,
        include_groups=False
    )
    .reset_index()
)


change_results["Mean_Temperature_Change_C"] = (
    analysis_df
    .groupby(
        "temperature_change_bin",
        observed=False
    )["temperature_change_1h"]
    .mean()
    .values
)


change_results.to_csv(
    f"{REPORTS_DIR}/lstm_v2_temperature_change_results.csv",
    index=False
)


# ============================================================
# SAVE FULL ERROR DATA
# ============================================================

analysis_df.to_csv(
    f"{REPORTS_DIR}/lstm_v2_temperature_error_analysis.csv",
    index=False
)


# ============================================================
# FIGURE 1 — ERROR VS ACTUAL TEMPERATURE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    analysis_df["actual_temperature"],
    analysis_df["error"],
    alpha=0.25,
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
    "Prediction Error (Actual - Predicted) (°C)"
)

plt.title(
    "LSTM v2 Prediction Error vs Actual Temperature"
)

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/lstm_v2_error_vs_actual_temperature.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 2 — SIGNED ERROR VS TEMPERATURE
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    analysis_df["actual_temperature"],
    analysis_df["error"],
    alpha=0.20,
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
    "Signed Error (°C)"
)

plt.title(
    "LSTM v2 Signed Prediction Error by Actual Temperature"
)

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/lstm_v2_signed_error_vs_actual_temperature.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 3 — MAE BY TEMPERATURE BIN
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    temperature_results["temperature_bin"].astype(str),
    temperature_results["MAE_C"]
)

plt.xlabel(
    "Actual Temperature Range"
)

plt.ylabel(
    "MAE (°C)"
)

plt.title(
    "LSTM v2 MAE by Actual Temperature Range"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/lstm_v2_mae_by_temperature_bin.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 4 — BIAS BY TEMPERATURE BIN
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    temperature_results["temperature_bin"].astype(str),
    temperature_results["Mean_Error_C"]
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
    "LSTM v2 Bias by Actual Temperature Range"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/lstm_v2_bias_by_temperature_bin.png",
    dpi=300
)

plt.close()


# ============================================================
# FIGURE 5 — MAE BY TEMPERATURE CHANGE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    change_results["temperature_change_bin"].astype(str),
    change_results["MAE_C"]
)

plt.xlabel(
    "One-Hour Actual Temperature Change"
)

plt.ylabel(
    "MAE (°C)"
)

plt.title(
    "LSTM v2 MAE by One-Hour Temperature Change"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    f"{FIGURES_DIR}/lstm_v2_mae_by_temperature_change.png",
    dpi=300
)

plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("LSTM v2 TEMPERATURE ERROR ANALYSIS")
print("=" * 75)

print("\nTEMPERATURE BIN RESULTS")
print(
    temperature_results.to_string(
        index=False
    )
)

print("\nTEMPERATURE QUANTILE RESULTS")
print(
    quantile_results.to_string(
        index=False
    )
)

print("\nTEMPERATURE CHANGE RESULTS")
print(
    change_results.to_string(
        index=False
    )
)

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print(
    "reports/lstm_v2_temperature_bin_results.csv"
)

print(
    "reports/lstm_v2_temperature_quantile_results.csv"
)

print(
    "reports/lstm_v2_temperature_change_results.csv"
)

print(
    "reports/lstm_v2_temperature_error_analysis.csv"
)

print(
    "reports/figures/lstm_v2_error_vs_actual_temperature.png"
)

print(
    "reports/figures/lstm_v2_signed_error_vs_actual_temperature.png"
)

print(
    "reports/figures/lstm_v2_mae_by_temperature_bin.png"
)

print(
    "reports/figures/lstm_v2_bias_by_temperature_bin.png"
)

print(
    "reports/figures/lstm_v2_mae_by_temperature_change.png"
)

print("=" * 75)