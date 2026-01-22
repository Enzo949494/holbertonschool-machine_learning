#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

data = np.load("btc_normalized.npz")
values = data["data"]

print(f"Loaded {len(values)} time steps")

# Subsample pour RAM
values = values[::20]  # Réduit à ~82k
n = len(values)
print(f"Subsampled to {n} steps")

WINDOW_SIZE = 1440
HORIZON = 60
BATCH_SIZE = 128

def make_windows(start, end):
    X, y = [], []
    for i in range(start, end - WINDOW_SIZE - HORIZON):
        X.append(values[i:i+WINDOW_SIZE])
        y.append(values[i+WINDOW_SIZE, 0])
    return np.array(X), np.array(y)

# Split
train_X, train_y = make_windows(0, int(0.7*n))
val_X, val_y = make_windows(int(0.7*n), int(0.85*n))
test_X, test_y = make_windows(int(0.85*n), n)

print(f"Train: {len(train_X)} | Val: {len(val_X)} | Test: {len(test_X)}")

# tf.data
train_ds = tf.data.Dataset.from_tensor_slices((train_X, train_y)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((val_X, val_y)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(BATCH_SIZE)

# Modèle
model = keras.Sequential([
    layers.LSTM(64, input_shape=(WINDOW_SIZE, 4)),
    layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("🚀 Training...")
model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stop])

loss = model.evaluate(test_ds)
print(f"MSE: {loss:.4f}")

model.save("btc_model.h5")
print("✅ DONE!")
