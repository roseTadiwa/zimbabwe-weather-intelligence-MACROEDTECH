import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. Load the model dataset
# ============================================================

DATA_PATH = "data/processed/zimbabwe_weather_model.csv"

df = pd.read_csv(DATA_PATH)

df["time"] = pd.to_datetime(df["time"])

print("Dataset shape:", df.shape)
print("Date range:", df["time"].min(), "to", df["time"].max())
print("Cities:", df["city"].unique())


# ============================================================
# 2. Sort data correctly
# ============================================================

df = df.sort_values(["city", "time"]).reset_index(drop=True)


# ============================================================
# 3. Define features and target
# ============================================================

FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "temperature_lag_1h",
    "temperature_lag_24h",
    "temperature_rolling_24h",
    "precipitation_lag_1h",
]

TARGET = "target_temperature_1h"


# ============================================================
# 4. Define chronological periods
# ============================================================

TRAIN_END = "2022-12-31 23:00:00"
VAL_END = "2024-12-31 23:00:00"


# ============================================================
# 5. Split data chronologically
# ============================================================

train_df = df[df["time"] <= TRAIN_END].copy()

val_df = df[
    (df["time"] > TRAIN_END)
    & (df["time"] <= VAL_END)
].copy()

test_df = df[df["time"] > VAL_END].copy()


print("\nChronological split:")
print("Training:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# ============================================================
# 6. Scale using TRAINING data only
# ============================================================

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

feature_scaler.fit(train_df[FEATURES])
target_scaler.fit(train_df[[TARGET]])


train_scaled = train_df.copy()
val_scaled = val_df.copy()
test_scaled = test_df.copy()

train_scaled[FEATURES] = feature_scaler.transform(
    train_df[FEATURES]
)

val_scaled[FEATURES] = feature_scaler.transform(
    val_df[FEATURES]
)

test_scaled[FEATURES] = feature_scaler.transform(
    test_df[FEATURES]
)


train_scaled[TARGET] = target_scaler.transform(
    train_df[[TARGET]]
)

val_scaled[TARGET] = target_scaler.transform(
    val_df[[TARGET]]
)

test_scaled[TARGET] = target_scaler.transform(
    test_df[[TARGET]]
)


# ============================================================
# 7. Create sequential LSTM datasets
# ============================================================

LOOKBACK = 24


def create_sequences(data, features, target, lookback=24):

    X = []
    y = []

    for city in data["city"].unique():

        city_data = data[data["city"] == city].copy()

        city_features = city_data[features].values
        city_target = city_data[target].values

        for i in range(lookback, len(city_data)):

            X.append(city_features[i - lookback:i])
            y.append(city_target[i])

    return np.array(X), np.array(y)


# ============================================================
# 8. Create sequences
# ============================================================

X_train, y_train = create_sequences(
    train_scaled,
    FEATURES,
    TARGET,
    LOOKBACK
)

X_val, y_val = create_sequences(
    val_scaled,
    FEATURES,
    TARGET,
    LOOKBACK
)

X_test, y_test = create_sequences(
    test_scaled,
    FEATURES,
    TARGET,
    LOOKBACK
)


# ============================================================
# 9. Create output directory
# ============================================================

LSTM_DATA_DIR = "data/processed/lstm_data"

os.makedirs(LSTM_DATA_DIR, exist_ok=True)


# ============================================================
# 10. Save sequences
# ============================================================

np.save(
    f"{LSTM_DATA_DIR}/X_train.npy",
    X_train
)

np.save(
    f"{LSTM_DATA_DIR}/y_train.npy",
    y_train
)

np.save(
    f"{LSTM_DATA_DIR}/X_val.npy",
    X_val
)

np.save(
    f"{LSTM_DATA_DIR}/y_val.npy",
    y_val
)

np.save(
    f"{LSTM_DATA_DIR}/X_test.npy",
    X_test
)

np.save(
    f"{LSTM_DATA_DIR}/y_test.npy",
    y_test
)


# ============================================================
# 11. Save scalers
# ============================================================

joblib.dump(
    feature_scaler,
    f"{LSTM_DATA_DIR}/feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    f"{LSTM_DATA_DIR}/target_scaler.pkl"
)


# ============================================================
# 12. Display final information
# ============================================================

print("\nLSTM sequence shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nNumber of features:", len(FEATURES))
print("Lookback hours:", LOOKBACK)

print("\nSaved LSTM data to:")
print(LSTM_DATA_DIR)

print("\nFiles saved:")
print("- X_train.npy")
print("- y_train.npy")
print("- X_val.npy")
print("- y_val.npy")
print("- X_test.npy")
print("- y_test.npy")
print("- feature_scaler.pkl")
print("- target_scaler.pkl")

print("\nLSTM data preparation completed successfully.")