import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "reports/xgboost_feature_importance.csv"
OUTPUT_PATH = "reports/figures/xgboost_feature_importance.png"


# ============================================================
# LOAD FEATURE IMPORTANCE
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("XGBOOST FEATURE IMPORTANCE")
print("=" * 70)

print(df)


# ============================================================
# SORT FEATURES
# ============================================================

df = df.sort_values(
    by=df.columns[-1],
    ascending=True
)


# ============================================================
# SELECT TOP FEATURES
# ============================================================

top_features = df.tail(10)


# ============================================================
# CREATE VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    top_features.iloc[:, 0],
    top_features.iloc[:, -1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.title(
    "XGBoost Top 10 Feature Importance"
)

plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nVisualization saved to:")
print(OUTPUT_PATH)