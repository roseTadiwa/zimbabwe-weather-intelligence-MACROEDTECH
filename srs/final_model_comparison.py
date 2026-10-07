import pandas as pd


# ============================================================
# PATHS
# ============================================================

PERSISTENCE_PATH = "reports/persistence_results.csv"
LINEAR_PATH = "reports/linear_regression_results.csv"
RF_PATH = "reports/random_forest_results.csv"
XGB_PATH = "reports/xgboost_results.csv"
LSTM_PATH = "reports/lstm_results.csv"
LSTM_V2_PATH = "reports/lstm_v2_results.csv"

OUTPUT_PATH = "reports/final_model_comparison.csv"


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_test_metrics(path):
    """
    Reads a model result CSV and extracts:
    Test MAE, Test RMSE and Test R2.

    Handles the different CSV formats used in the project.
    """

    df = pd.read_csv(path)

    print(f"\nLoaded: {path}")
    print(f"Columns: {df.columns.tolist()}")

    # --------------------------------------------------------
    # FORMAT 1
    # Test_MAE / Test_RMSE / Test_R2
    # --------------------------------------------------------

    if all(
        column in df.columns
        for column in [
            "Test_MAE",
            "Test_RMSE",
            "Test_R2"
        ]
    ):

        row = df.iloc[0]

        return (
            float(row["Test_MAE"]),
            float(row["Test_RMSE"]),
            float(row["Test_R2"])
        )

    # --------------------------------------------------------
    # FORMAT 2
    # test_mae / test_rmse / test_r2
    # --------------------------------------------------------

    if all(
        column in df.columns
        for column in [
            "test_mae",
            "test_rmse",
            "test_r2"
        ]
    ):

        row = df.iloc[0]

        return (
            float(row["test_mae"]),
            float(row["test_rmse"]),
            float(row["test_r2"])
        )

    # --------------------------------------------------------
    # FORMAT 3
    # Dataset / MAE_C / RMSE_C / R2
    # --------------------------------------------------------

    if all(
        column in df.columns
        for column in [
            "Dataset",
            "MAE_C",
            "RMSE_C",
            "R2"
        ]
    ):

        test_rows = df[
            df["Dataset"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "test"
        ]

        if test_rows.empty:
            raise ValueError(
                f"No Test row found in {path}"
            )

        row = test_rows.iloc[0]

        return (
            float(row["MAE_C"]),
            float(row["RMSE_C"]),
            float(row["R2"])
        )

    # --------------------------------------------------------
    # FORMAT 4
    # model / validation_mae / validation_rmse /
    # validation_r2 / test_mae / test_rmse / test_r2
    # --------------------------------------------------------

    raise ValueError(
        f"Unrecognized result format in {path}.\n"
        f"Columns found: {df.columns.tolist()}"
    )


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

print("Loading model results...")


persistence_mae, persistence_rmse, persistence_r2 = (
    get_test_metrics(PERSISTENCE_PATH)
)

linear_mae, linear_rmse, linear_r2 = (
    get_test_metrics(LINEAR_PATH)
)

rf_mae, rf_rmse, rf_r2 = (
    get_test_metrics(RF_PATH)
)

xgb_mae, xgb_rmse, xgb_r2 = (
    get_test_metrics(XGB_PATH)
)

lstm_mae, lstm_rmse, lstm_r2 = (
    get_test_metrics(LSTM_PATH)
)

lstm_v2_mae, lstm_v2_rmse, lstm_v2_r2 = (
    get_test_metrics(LSTM_V2_PATH)
)


# ============================================================
# CREATE FINAL COMPARISON TABLE
# ============================================================

comparison = pd.DataFrame([
    {
        "Model": "Persistence",
        "Test_MAE_C": persistence_mae,
        "Test_RMSE_C": persistence_rmse,
        "Test_R2": persistence_r2
    },
    {
        "Model": "Linear Regression",
        "Test_MAE_C": linear_mae,
        "Test_RMSE_C": linear_rmse,
        "Test_R2": linear_r2
    },
    {
        "Model": "Random Forest",
        "Test_MAE_C": rf_mae,
        "Test_RMSE_C": rf_rmse,
        "Test_R2": rf_r2
    },
    {
        "Model": "XGBoost",
        "Test_MAE_C": xgb_mae,
        "Test_RMSE_C": xgb_rmse,
        "Test_R2": xgb_r2
    },
    {
        "Model": "Baseline LSTM",
        "Test_MAE_C": lstm_mae,
        "Test_RMSE_C": lstm_rmse,
        "Test_R2": lstm_r2
    },
    {
        "Model": "LSTM v2",
        "Test_MAE_C": lstm_v2_mae,
        "Test_RMSE_C": lstm_v2_rmse,
        "Test_R2": lstm_v2_r2
    }
])


# ============================================================
# ROUND RESULTS
# ============================================================

comparison[
    [
        "Test_MAE_C",
        "Test_RMSE_C",
        "Test_R2"
    ]
] = comparison[
    [
        "Test_MAE_C",
        "Test_RMSE_C",
        "Test_R2"
    ]
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

print("\n")
print("FINAL MODEL COMPARISON — 2025 TEST SET")
print("=" * 75)

print(
    comparison.to_string(
        index=False
    )
)

print("=" * 75)

print(
    f"\nResults saved to: {OUTPUT_PATH}"
)