import os
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

RESULTS_FILES = {
    "Persistence": "reports/persistence_results.csv",
    "Linear Regression": "reports/linear_regression_results.csv",
    "Random Forest": "reports/random_forest_results.csv",
    "XGBoost": "reports/xgboost_results.csv",
    "LSTM": "reports/lstm_results.csv",
}

OUTPUT_PATH = "reports/model_comparison.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

print("Loading model results...")

all_results = []


for model_name, file_path in RESULTS_FILES.items():

    print(f"\nReading: {file_path}")

    if not os.path.exists(file_path):
        print(f"WARNING: File not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    print(f"Available columns: {list(df.columns)}")

    if df.empty:
        print(f"WARNING: {file_path} is empty.")
        continue


    # ========================================================
    # LSTM RESULTS
    # ========================================================

    if model_name == "LSTM":

        # Select the TEST row specifically.
        test_rows = df[
            df["Dataset"].astype(str).str.lower() == "test"
        ]

        if test_rows.empty:
            print(
                "ERROR: Test row not found in LSTM results."
            )
            continue

        row = test_rows.iloc[0]

        mae = float(row["MAE_C"])
        rmse = float(row["RMSE_C"])
        r2 = float(row["R2"])


    # ========================================================
    # OTHER MODELS
    # ========================================================

    else:

        # Find test metric columns regardless of capitalization.
        columns_lower = {
            column.lower(): column
            for column in df.columns
        }

        mae_column = columns_lower.get("test_mae")
        rmse_column = columns_lower.get("test_rmse")
        r2_column = columns_lower.get("test_r2")

        if not mae_column:
            print(
                f"ERROR: Test MAE not found in {file_path}"
            )
            continue

        if not rmse_column:
            print(
                f"ERROR: Test RMSE not found in {file_path}"
            )
            continue

        if not r2_column:
            print(
                f"ERROR: Test R² not found in {file_path}"
            )
            continue

        row = df.iloc[0]

        mae = float(row[mae_column])
        rmse = float(row[rmse_column])
        r2 = float(row[r2_column])


    # ========================================================
    # STORE RESULTS
    # ========================================================

    all_results.append({
        "Model": model_name,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2
    })

    print(f"Successfully loaded {model_name}")


# ============================================================
# CHECK RESULTS
# ============================================================

if not all_results:
    raise ValueError(
        "No model results could be loaded."
    )


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame(
    all_results
)


# ============================================================
# MODEL ORDER
# ============================================================

model_order = [
    "Persistence",
    "Linear Regression",
    "Random Forest",
    "XGBoost",
    "LSTM"
]

comparison["Model"] = pd.Categorical(
    comparison["Model"],
    categories=model_order,
    ordered=True
)

comparison = comparison.sort_values(
    "Model"
).reset_index(drop=True)


# ============================================================
# ROUND METRICS
# ============================================================

comparison["Test_MAE"] = comparison[
    "Test_MAE"
].round(4)

comparison["Test_RMSE"] = comparison[
    "Test_RMSE"
].round(4)

comparison["Test_R2"] = comparison[
    "Test_R2"
].round(4)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL CONSOLIDATED MODEL COMPARISON")
print("=" * 75)

print(
    comparison.to_string(index=False)
)

print("=" * 75)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

comparison.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nComparison saved to: {OUTPUT_PATH}"
)

print(
    "\nModel comparison completed successfully."
)