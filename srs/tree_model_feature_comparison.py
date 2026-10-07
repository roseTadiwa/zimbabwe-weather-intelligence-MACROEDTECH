import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

RF_PATH = "reports/random_forest_feature_importance.csv"
XGB_PATH = "reports/xgboost_feature_importance.csv"

OUTPUT_PATH = (
    "reports/figures/"
    "tree_model_feature_importance_comparison.png"
)


# ============================================================
# LOAD DATA
# ============================================================

rf = pd.read_csv(RF_PATH)
xgb = pd.read_csv(XGB_PATH)


# Rename columns consistently
rf.columns = ["feature", "random_forest_importance"]
xgb.columns = ["feature", "xgboost_importance"]


# ============================================================
# MERGE
# ============================================================

comparison = pd.merge(
    rf,
    xgb,
    on="feature",
    how="outer"
).fillna(0)


# ============================================================
# SELECT TOP FEATURES
# ============================================================

comparison["average_importance"] = (
    comparison["random_forest_importance"]
    + comparison["xgboost_importance"]
) / 2

top_features = (
    comparison
    .sort_values(
        "average_importance",
        ascending=False
    )
    .head(10)
    .sort_values(
        "average_importance",
        ascending=True
    )
)


# ============================================================
# DISPLAY
# ============================================================

print("TREE MODEL FEATURE IMPORTANCE COMPARISON")
print("=" * 80)

print(
    top_features[
        [
            "feature",
            "random_forest_importance",
            "xgboost_importance"
        ]
    ].to_string(index=False)
)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(11, 7))

y_positions = range(len(top_features))

plt.barh(
    [y - 0.18 for y in y_positions],
    top_features["random_forest_importance"],
    height=0.35,
    label="Random Forest"
)

plt.barh(
    [y + 0.18 for y in y_positions],
    top_features["xgboost_importance"],
    height=0.35,
    label="XGBoost"
)

plt.yticks(
    y_positions,
    top_features["feature"]
)

plt.xlabel("Feature Importance")

plt.ylabel("Feature")

plt.title(
    "Random Forest vs XGBoost Feature Importance"
)

plt.legend()

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nVisualization saved to:")
print(OUTPUT_PATH)