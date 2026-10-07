import os
import numpy as np
import tensorflow as tf
import keras


# ============================================================
# PATHS
# ============================================================

LSTM_DATA_DIR = "data/processed/lstm_data_v2"

MODEL_PATH = "models/lstm_weather_model_v2.keras"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading improved LSTM data...")

X_train = np.load(
    f"{LSTM_DATA_DIR}/X_train.npy"
)

y_train = np.load(
    f"{LSTM_DATA_DIR}/y_train.npy"
)

X_val = np.load(
    f"{LSTM_DATA_DIR}/X_val.npy"
)

y_val = np.load(
    f"{LSTM_DATA_DIR}/y_val.npy"
)

X_test = np.load(
    f"{LSTM_DATA_DIR}/X_test.npy"
)

y_test = np.load(
    f"{LSTM_DATA_DIR}/y_test.npy"
)


print("\nDataset shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# BUILD IMPROVED LSTM MODEL
# ============================================================

print("\nBuilding improved LSTM model...")

model = keras.Sequential([
    keras.layers.Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    keras.layers.LSTM(
        64,
        return_sequences=True
    ),

    keras.layers.Dropout(
        0.2
    ),

    keras.layers.LSTM(
        32
    ),

    keras.layers.Dropout(
        0.2
    ),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dense(
        1
    )
])


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="mse",

    metrics=[
        keras.metrics.MeanAbsoluteError(
            name="mae"
        )
    ]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=256,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

model.save(
    MODEL_PATH
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED LSTM TRAINING COMPLETE")
print("=" * 70)

print(
    f"Total epochs trained: "
    f"{len(history.history['loss'])}"
)

print(
    f"Best validation loss: "
    f"{min(history.history['val_loss']):.6f}"
)

print(
    f"Best validation MAE: "
    f"{min(history.history['val_mae']):.6f}"
)

print(
    f"\nModel saved to: "
    f"{MODEL_PATH}"
)

print("=" * 70)