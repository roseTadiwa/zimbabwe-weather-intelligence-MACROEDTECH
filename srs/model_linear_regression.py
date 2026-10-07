import pandas as pd
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/processed/model_data")

TRAIN_FILE = DATA_DIR / "train.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"
TEST_FILE = DATA_DIR / "test.csv"


# ============================================================
# Load datasets
# ============================================================

print("=" * 60)
print("Zimbabwe Weather Intelligence")
print("Linear Regression Baseline Model")
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
# Prepare X and y
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
# Train Linear Regression model
# ============================================================

print("\nTraining Linear Regression model...")

model = LinearRegression()

model.fit(X_train, y_train)

print("Model training completed successfully.")


# ============================================================
# Validation predictions
# ============================================================

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
# Save model results
# ============================================================

results = pd.DataFrame({
    "model": ["Linear Regression"],
    "validation_mae": [validation_mae],
    "validation_rmse": [validation_rmse],
    "validation_r2": [validation_r2],
    "test_mae": [test_mae],
    "test_rmse": [test_rmse],
    "test_r2": [test_r2]
})

results_path = Path("reports/linear_regression_results.csv")
results_path.parent.mkdir(parents=True, exist_ok=True)

results.to_csv(results_path, index=False)

print("\nResults saved to:")
print(results_path)

print("\nLinear Regression baseline completed successfully.")