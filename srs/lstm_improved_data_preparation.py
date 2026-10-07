import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# ============================================================
# PATHS
# ============================================================

MODEL_DATA_PATH = "data/processed/zimbabwe_weather_model.csv"

OUTPUT_DIR = "data/processed/lstm_data_v2"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

LOOKBACK = 24

TRAIN_END = "2022-12-31 23:00:00"
VAL_END = "2024-12-31 23:00:00"


# ============================================================
# FEATURES
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

    # New feature
    "temperature_change_1h",
]

TARGET = "target_temperature_1h"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading model dataset...")

df = pd.read_csv(
    MODEL_DATA_PATH,
    parse_dates=["time"]
)

print(f"Dataset shape: {df.shape}")
print(
    f"Date range: {df['time'].min()} "
    f"to {df['time'].max()}"
)
print(f"Cities: {sorted(df['city'].unique())}")


# ============================================================
# CREATE NEW TEMPERATURE-DYNAMICS FEATURE
# ============================================================

print("\nCreating temperature_change_1h...")

df["temperature_change_1h"] = (
    df["temperature_2m"]
    - df["temperature_lag_1h"]
)


# ============================================================
# REMOVE ROWS WITH MISSING VALUES
# ============================================================

required_columns = FEATURES + [TARGET]

df = df.dropna(
    subset=required_columns
).copy()

df = df.sort_values(
    ["city", "time"]
).reset_index(drop=True)


print(
    f"Dataset shape after removing missing values: "
    f"{df.shape}"
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train_df = df[
    df["time"] <= TRAIN_END
].copy()

val_df = df[
    (df["time"] > TRAIN_END)
    & (df["time"] <= VAL_END)
].copy()

test_df = df[
    df["time"] > VAL_END
].copy()


print("\nChronological split:")
print(f"Training:   {train_df.shape}")
print(f"Validation: {val_df.shape}")
print(f"Test:       {test_df.shape}")


# ============================================================
# SCALE FEATURES
# ============================================================

print("\nFitting scalers using training data only...")

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()


feature_scaler.fit(
    train_df[FEATURES]
)

target_scaler.fit(
    train_df[[TARGET]]
)


# ============================================================
# TRANSFORM DATA
# ============================================================

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
).ravel()

val_scaled[TARGET] = target_scaler.transform(
    val_df[[TARGET]]
).ravel()

test_scaled[TARGET] = target_scaler.transform(
    test_df[[TARGET]]
).ravel()


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_sequences(data):
    """
    Create chronological sequences separately for each city.

    Each sequence contains LOOKBACK hours of historical
    observations and predicts the next-hour temperature.
    """

    X = []
    y = []

    for city in sorted(data["city"].unique()):

        city_data = data[
            data["city"] == city
        ].sort_values("time").reset_index(drop=True)

        feature_values = city_data[
            FEATURES
        ].values

        target_values = city_data[
            TARGET
        ].values

        for i in range(
            LOOKBACK,
            len(city_data)
        ):
            X.append(
                feature_values[
                    i - LOOKBACK:i
                ]
            )

            y.append(
                target_values[i]
            )

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32)
    )


# ============================================================
# CREATE SEQUENCES
# ============================================================

print("\nCreating LSTM sequences...")

X_train, y_train = create_sequences(
    train_scaled
)

X_val, y_val = create_sequences(
    val_scaled
)

X_test, y_test = create_sequences(
    test_scaled
)


# ============================================================
# SAVE ARRAYS
# ============================================================

np.save(
    f"{OUTPUT_DIR}/X_train.npy",
    X_train
)

np.save(
    f"{OUTPUT_DIR}/y_train.npy",
    y_train
)

np.save(
    f"{OUTPUT_DIR}/X_val.npy",
    X_val
)

np.save(
    f"{OUTPUT_DIR}/y_val.npy",
    y_val
)

np.save(
    f"{OUTPUT_DIR}/X_test.npy",
    X_test
)

np.save(
    f"{OUTPUT_DIR}/y_test.npy",
    y_test
)


# ============================================================
# SAVE SCALERS
# ============================================================

joblib.dump(
    feature_scaler,
    f"{OUTPUT_DIR}/feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    f"{OUTPUT_DIR}/target_scaler.pkl"
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

with open(
    f"{OUTPUT_DIR}/features.txt",
    "w",
    encoding="utf-8"
) as file:

    for feature in FEATURES:
        file.write(feature + "\n")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED LSTM DATA PREPARATION COMPLETE")
print("=" * 70)

print(f"Number of features: {len(FEATURES)}")
print(f"Lookback hours: {LOOKBACK}")

print(f"\nX_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print(f"\nX_val: {X_val.shape}")
print(f"y_val: {y_val.shape}")

print(f"\nX_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")

print("\nFeatures:")
for feature in FEATURES:
    print(f" - {feature}")

print(
    f"\nSaved to: {OUTPUT_DIR}"
)

print("=" * 70)