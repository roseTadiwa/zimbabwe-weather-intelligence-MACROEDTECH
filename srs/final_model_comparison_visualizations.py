import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "reports/final_model_comparison.csv"

MAE_OUTPUT = (
    "reports/figures/final_model_comparison_mae.png"
)

RMSE_OUTPUT = (
    "reports/figures/final_model_comparison_rmse.png"
)

R2_OUTPUT = (
    "reports/figures/final_model_comparison_r2.png"
)


# ============================================================
# LOAD FINAL COMPARISON DATA
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("FINAL MODEL COMPARISON VISUALISATIONS")
print("=" * 75)

print(df)


# ============================================================
# MODEL ORDER
# ============================================================

model_order = [
    "Persistence",
    "Linear Regression",
    "Random Forest",
    "XGBoost",
    "Baseline LSTM",
    "LSTM v2"
]

df["Model"] = pd.Categorical(
    df["Model"],
    categories=model_order,
    ordered=True
)

df = df.sort_values("Model")


# ============================================================
# MAE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Test_MAE_C"]
)

plt.ylabel("Test MAE (°C)")
plt.xlabel("Model")

plt.title(
    "Final Model Comparison — Test MAE"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    MAE_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RMSE
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Test_RMSE_C"]
)

plt.ylabel("Test RMSE (°C)")
plt.xlabel("Model")

plt.title(
    "Final Model Comparison — Test RMSE"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    RMSE_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# R²
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Test_R2"]
)

plt.ylabel("Test R²")
plt.xlabel("Model")

plt.title(
    "Final Model Comparison — Test R²"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    R2_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CONFIRM OUTPUTS
# ============================================================

print("\nVisualisations saved to:")

print(MAE_OUTPUT)
print(RMSE_OUTPUT)
print(R2_OUTPUT)