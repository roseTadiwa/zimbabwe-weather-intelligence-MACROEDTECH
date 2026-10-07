import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASELINE_PATH = "reports/lstm_city_results.csv"
V2_PATH = "reports/lstm_v2_city_results.csv"

OUTPUT_PATH = "reports/lstm_city_comparison.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

print("Loading baseline LSTM city results...")
baseline = pd.read_csv(BASELINE_PATH)

print("Loading LSTM v2 city results...")
v2 = pd.read_csv(V2_PATH)


# ============================================================
# SELECT REQUIRED COLUMNS
# ============================================================

baseline = baseline[
    [
        "City",
        "Test_Sequences",
        "MAE_C",
        "RMSE_C",
        "R2",
        "Mean_Error_C"
    ]
].copy()

v2 = v2[
    [
        "City",
        "Test_Sequences",
        "MAE_C",
        "RMSE_C",
        "R2",
        "Mean_Error_C"
    ]
].copy()


# ============================================================
# RENAME COLUMNS
# ============================================================

baseline = baseline.rename(
    columns={
        "Test_Sequences": "Baseline_Sequences",
        "MAE_C": "Baseline_MAE_C",
        "RMSE_C": "Baseline_RMSE_C",
        "R2": "Baseline_R2",
        "Mean_Error_C": "Baseline_Mean_Error_C"
    }
)

v2 = v2.rename(
    columns={
        "Test_Sequences": "V2_Sequences",
        "MAE_C": "V2_MAE_C",
        "RMSE_C": "V2_RMSE_C",
        "R2": "V2_R2",
        "Mean_Error_C": "V2_Mean_Error_C"
    }
)


# ============================================================
# MERGE
# ============================================================

comparison = pd.merge(
    baseline,
    v2,
    on="City",
    how="inner"
)


# ============================================================
# CALCULATE IMPROVEMENTS
# ============================================================

comparison["MAE_Improvement_C"] = (
    comparison["Baseline_MAE_C"]
    - comparison["V2_MAE_C"]
)

comparison["RMSE_Improvement_C"] = (
    comparison["Baseline_RMSE_C"]
    - comparison["V2_RMSE_C"]
)

comparison["R2_Improvement"] = (
    comparison["V2_R2"]
    - comparison["Baseline_R2"]
)


# ============================================================
# PERCENTAGE IMPROVEMENTS
# ============================================================

comparison["MAE_Improvement_Percent"] = (
    comparison["MAE_Improvement_C"]
    / comparison["Baseline_MAE_C"]
) * 100

comparison["RMSE_Improvement_Percent"] = (
    comparison["RMSE_Improvement_C"]
    / comparison["Baseline_RMSE_C"]
) * 100


# ============================================================
# BIAS CHANGE
# ============================================================

comparison["Mean_Error_Change_C"] = (
    comparison["V2_Mean_Error_C"]
    - comparison["Baseline_Mean_Error_C"]
)


# ============================================================
# ROUND VALUES
# ============================================================

numeric_columns = comparison.select_dtypes(
    include="number"
).columns

comparison[numeric_columns] = comparison[
    numeric_columns
].round(4)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nBASELINE LSTM VS LSTM v2 — CITY-LEVEL COMPARISON")
print("=" * 110)

print(
    comparison.to_string(
        index=False
    )
)

print("=" * 110)

print(
    f"\nResults saved to: {OUTPUT_PATH}"
)