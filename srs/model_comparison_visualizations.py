import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FILE PATHS
# ============================================================

INPUT_PATH = "reports/model_comparison.csv"

FIGURES_DIR = "reports/figures"


# ============================================================
# CREATE FIGURES DIRECTORY
# ============================================================

os.makedirs(
    FIGURES_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

print("Loading model comparison data...")

df = pd.read_csv(INPUT_PATH)

print("\nModel comparison data:")
print(df)


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

df["Model"] = pd.Categorical(
    df["Model"],
    categories=model_order,
    ordered=True
)

df = df.sort_values(
    "Model"
)


# ============================================================
# 1. MAE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    df["Model"],
    df["Test_MAE"]
)

plt.title(
    "Test Mean Absolute Error (MAE) by Model"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "MAE (°C)"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

mae_path = os.path.join(
    FIGURES_DIR,
    "model_comparison_mae.png"
)

plt.savefig(
    mae_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {mae_path}"
)


# ============================================================
# 2. RMSE
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    df["Model"],
    df["Test_RMSE"]
)

plt.title(
    "Test Root Mean Square Error (RMSE) by Model"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "RMSE (°C)"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

rmse_path = os.path.join(
    FIGURES_DIR,
    "model_comparison_rmse.png"
)

plt.savefig(
    rmse_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {rmse_path}"
)


# ============================================================
# 3. R²
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    df["Model"],
    df["Test_R2"]
)

plt.title(
    "Test R² by Model"
)

plt.xlabel(
    "Model"
)

plt.ylabel(
    "R²"
)

plt.xticks(
    rotation=20
)

plt.tight_layout()

r2_path = os.path.join(
    FIGURES_DIR,
    "model_comparison_r2.png"
)

plt.savefig(
    r2_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {r2_path}"
)


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON VISUALISATIONS COMPLETED")
print("=" * 70)

print("\nFigures created:")

print(
    "1. reports/figures/model_comparison_mae.png"
)

print(
    "2. reports/figures/model_comparison_rmse.png"
)

print(
    "3. reports/figures/model_comparison_r2.png"
)

print("=" * 70)