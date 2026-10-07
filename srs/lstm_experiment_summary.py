import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASELINE_RESULTS = "reports/lstm_results.csv"
V2_RESULTS = "reports/lstm_v2_results.csv"

OUTPUT_PATH = "reports/lstm_experiment_summary.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

print("Loading baseline LSTM results...")
baseline = pd.read_csv(BASELINE_RESULTS)

print("Loading LSTM v2 results...")
v2 = pd.read_csv(V2_RESULTS)


# ============================================================
# DISPLAY ORIGINAL RESULTS
# ============================================================

print("\nBaseline LSTM results:")
print(baseline.to_string(index=False))

print("\nLSTM v2 results:")
print(v2.to_string(index=False))


# ============================================================
# EXTRACT BASELINE TEST RESULTS
# ============================================================

baseline_test = baseline[
    baseline["Dataset"] == "Test"
].iloc[0]

baseline_mae = baseline_test["MAE_C"]
baseline_rmse = baseline_test["RMSE_C"]
baseline_r2 = baseline_test["R2"]


# ============================================================
# EXTRACT LSTM v2 TEST RESULTS
# ============================================================

v2_test = v2.iloc[0]

v2_mae = v2_test["Test_MAE"]
v2_rmse = v2_test["Test_RMSE"]
v2_r2 = v2_test["Test_R2"]


# ============================================================
# CALCULATE IMPROVEMENTS
# ============================================================

mae_improvement = baseline_mae - v2_mae

rmse_improvement = baseline_rmse - v2_rmse

r2_improvement = v2_r2 - baseline_r2

mae_improvement_percent = (
    mae_improvement / baseline_mae
) * 100

rmse_improvement_percent = (
    rmse_improvement / baseline_rmse
) * 100


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

summary = pd.DataFrame([
    {
        "Experiment": "Baseline LSTM",
        "Features": 15,
        "Test_MAE_C": baseline_mae,
        "Test_RMSE_C": baseline_rmse,
        "Test_R2": baseline_r2,
        "MAE_Improvement_C": 0,
        "RMSE_Improvement_C": 0,
        "R2_Improvement": 0,
        "MAE_Improvement_Percent": 0,
        "RMSE_Improvement_Percent": 0
    },
    {
        "Experiment": "LSTM v2",
        "Features": 16,
        "Test_MAE_C": v2_mae,
        "Test_RMSE_C": v2_rmse,
        "Test_R2": v2_r2,
        "MAE_Improvement_C": mae_improvement,
        "RMSE_Improvement_C": rmse_improvement,
        "R2_Improvement": r2_improvement,
        "MAE_Improvement_Percent": mae_improvement_percent,
        "RMSE_Improvement_Percent": rmse_improvement_percent
    }
])


# ============================================================
# ROUND VALUES
# ============================================================

numeric_columns = summary.select_dtypes(
    include="number"
).columns

summary[numeric_columns] = summary[
    numeric_columns
].round(4)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY FINAL SUMMARY
# ============================================================

print("\nLSTM EXPERIMENT SUMMARY")
print("=" * 110)

print(
    summary.to_string(
        index=False
    )
)

print("=" * 110)

print("\nOverall improvement from Baseline LSTM to LSTM v2:")

print(
    f"MAE:  {mae_improvement:.4f} °C "
    f"({mae_improvement_percent:.2f}% reduction)"
)

print(
    f"RMSE: {rmse_improvement:.4f} °C "
    f"({rmse_improvement_percent:.2f}% reduction)"
)

print(
    f"R²:   +{r2_improvement:.4f}"
)

print(
    f"\nSummary saved to: {OUTPUT_PATH}"
)