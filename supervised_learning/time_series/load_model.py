#!/usr/bin/env python3
"""
Load a trained Bitcoin price prediction model and make predictions.

This module loads a pre-trained LSTM model and uses it to predict Bitcoin
prices at various points in the dataset. The script:

1. Loads the trained model from btc_model.h5
2. Loads preprocessed normalized Bitcoin data
3. Makes predictions at multiple timesteps
4. Denormalizes predictions to display actual USD prices

The model uses the closing price normalization parameters to convert
predicted normalized values back to real Bitcoin prices in USD.

Usage:
    $ python3 load_model.py

Output:
    - Predictions at sample indices from the dataset
    - Final prediction using the last 24 hours of data
"""

import numpy as np
from tensorflow import keras

# Load the trained LSTM model (without compiling for faster loading)
print("Chargement du modèle...")
model = keras.models.load_model('btc_model.h5', compile=False)
print("✅ Modèle chargé!")

# Load preprocessed normalized data and normalization parameters
data = np.load('btc_normalized.npz')
values = data['data']  # Normalized features
scaler_mean = data['mean']  # Mean used for normalization
scaler_std = data['std']  # Standard deviation used for normalization

print(f"\nDonnées: {len(values)} timesteps")

# Test predictions at different points in the dataset
test_indices = [100, 1000, 5000, 10000, 50000, len(values)-1440]

print("\n🔮 Predictions at various dataset points:")
for idx in test_indices:
    if idx >= 0 and idx + 1440 <= len(values):
        # Extract 24-hour window and reshape for model input
        window = values[idx:idx+1440].reshape(1, 1440, 4)
        # Get normalized prediction
        pred_norm = model.predict(window, verbose=0)[0][0]
        # Denormalize using mean and std from training data
        pred_real = pred_norm * scaler_std[0] + scaler_mean[0]
        print(f"Index {idx}: ${pred_real:.2f}")
    else:
        print(f"Index {idx}: Skip (out of bounds)")

# Use the most recent 1440 timesteps (24 hours) for the final prediction
last_window = values[-1440:].reshape(1, 1440, 4)

# Generate normalized prediction
prediction_normalized = model.predict(last_window)[0][0]
print(f"\nPrédiction normalisée: {prediction_normalized:.6f}")

# Denormalize the prediction to get the real price in USD
# Using: real_value = normalized_value * std + mean
prediction_real = prediction_normalized * scaler_std[0] + scaler_mean[0]
print(f"\n💰 Final BTC Price Prediction (USD): ${prediction_real:.2f}")
