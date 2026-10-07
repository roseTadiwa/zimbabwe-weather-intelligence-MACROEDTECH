import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/processed/model_data")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"

MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "xgboost_weather_model.json"


# ============================================================
# Load datasets
# ============================================================

print("=" * 60)
print("Zimbabwe Weather Intelligence")
print("XGBoost Regression Model")
print("=" * 60)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)
test = pd.read_csv(TEST_FILE)

print("\nDatasets loaded successfully.")
print("Training shape:", train.shape)
print("Validation shape:", validation.shape)
print("Test shape:", test.shape)


# ============================================================
# Define features and target
# ============================================================

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "year",
    "month",
    "day",
    "hour",
    "day_of_year",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "temperature_lag_1h",
    "temperature_lag_24h",
    "temperature_rolling_24h",
    "precipitation_lag_1h"
]

TARGET = "target_temperature_1h"


# ============================================================
# Prepare data
# ============================================================

X_train = train[FEATURES]
y_train = train[TARGET]

X_validation = validation[FEATURES]
y_validation = validation[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

print("\nNumber of features:", len(FEATURES))
print("Target:", TARGET)


# ============================================================
# Train XGBoost
# ============================================================

print("\nTraining XGBoost model...")

model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed successfully.")


# ============================================================
# Save trained model
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model.save_model(MODEL_FILE)

print("\nTrained model saved to:")
print(MODEL_FILE)


# ============================================================
# Validation predictions
# ============================================================

print("\nGenerating validation predictions...")

validation_predictions = model.predict(X_validation)

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions
)

validation_rmse = mean_squared_error(
    y_validation,
    validation_predictions
) ** 0.5

validation_r2 = r2_score(
    y_validation,
    validation_predictions
)


# ============================================================
# Test predictions
# ============================================================

print("Generating test predictions...")

test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = mean_squared_error(
    y_test,
    test_predictions
) ** 0.5

test_r2 = r2_score(
    y_test,
    test_predictions
)


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("\nValidation Performance:")
print(f"  MAE  : {validation_mae:.4f} °C")
print(f"  RMSE : {validation_rmse:.4f} °C")
print(f"  R²   : {validation_r2:.4f}")

print("\nTest Performance:")
print(f"  MAE  : {test_mae:.4f} °C")
print(f"  RMSE : {test_rmse:.4f} °C")
print(f"  R²   : {test_r2:.4f}")


# ============================================================
# Feature importance
# ============================================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values(
    by="importance",
    ascending=False
)

print("\nTop 10 Feature Importances:")
print(importance.head(10).to_string(index=False))


# ============================================================
# Save results
# ============================================================

results = pd.DataFrame({
    "model": ["XGBoost"],
    "validation_mae": [validation_mae],
    "validation_rmse": [validation_rmse],
    "validation_r2": [validation_r2],
    "test_mae": [test_mae],
    "test_rmse": [test_rmse],
    "test_r2": [test_r2]
})

results_path = Path("reports/xgboost_results.csv")
results_path.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(results_path, index=False)

importance_path = Path("reports/xgboost_feature_importance.csv")
importance.to_csv(importance_path, index=False)

print("\nResults saved to:")
print(results_path)

print("Feature importance saved to:")
print(importance_path)

print("\nXGBoost model completed successfully.")