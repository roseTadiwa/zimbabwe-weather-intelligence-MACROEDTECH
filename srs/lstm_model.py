import os
import numpy as np
import tensorflow as tf
import keras


# ============================================================
# 1. Reproducibility
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 2. Paths
# ============================================================

DATA_DIR = "data/processed/lstm_data"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 3. Load prepared LSTM data
# ============================================================

X_train = np.load(f"{DATA_DIR}/X_train.npy")
y_train = np.load(f"{DATA_DIR}/y_train.npy")

X_val = np.load(f"{DATA_DIR}/X_val.npy")
y_val = np.load(f"{DATA_DIR}/y_val.npy")


print("Training data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nValidation data:")
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)


# ============================================================
# 4. Define LSTM model architecture
# ============================================================

model = keras.Sequential([
    
    keras.layers.Input(
        shape=(X_train.shape[1], X_train.shape[2])
    ),

    keras.layers.LSTM(
        64,
        return_sequences=True
    ),

    keras.layers.Dropout(0.2),

    keras.layers.LSTM(
        32
    ),

    keras.layers.Dropout(0.2),

    keras.layers.Dense(
        16,
        activation="relu"
    ),

    keras.layers.Dense(
        1
    )
])


# ============================================================
# 5. Compile model
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=[
        keras.metrics.MeanAbsoluteError(name="mae")
    ]
)


# ============================================================
# 6. Display model architecture
# ============================================================

print("\nLSTM model architecture:")
model.summary()


# ============================================================
# 7. Define callbacks
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
# 8. Train model
# ============================================================

print("\nStarting LSTM training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=256,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# 9. Save trained model
# ============================================================

MODEL_PATH = f"{MODEL_DIR}/lstm_weather_model.keras"

model.save(MODEL_PATH)


# ============================================================
# 10. Final information
# ============================================================

print("\nLSTM training completed successfully.")

print("Model saved to:")
print(MODEL_PATH)

print("\nEpochs completed:", len(history.history["loss"]))