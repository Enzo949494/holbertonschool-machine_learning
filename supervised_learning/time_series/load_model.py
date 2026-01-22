#!/usr/bin/env python3
import numpy as np
from tensorflow import keras

print("Chargement du modèle...")
model = keras.models.load_model('btc_model.h5', compile=False)
print("✅ Modèle chargé!")

# Charger les données normalisées
data = np.load('btc_normalized.npz')
values = data['data']
scaler_mean = data['mean']
scaler_std = data['std']

print(f"\nDonnées: {len(values)} timesteps")

# Tester sur différents points du dataset
test_indices = [100, 1000, 5000, 10000, 50000, len(values)-1440]

for idx in test_indices:
    if idx >= 0 and idx + 1440 <= len(values):
        window = values[idx:idx+1440].reshape(1, 1440, 4)
        pred_norm = model.predict(window, verbose=0)[0][0]
        pred_real = pred_norm * scaler_std[0] + scaler_mean[0]
        print(f"Index {idx}: ${pred_real:.2f}")
    else:
        print(f"Index {idx}: Skip (out of bounds)")

# Prendre les 1440 derniers timesteps (24 heures)
last_window = values[-1440:].reshape(1, 1440, 4)

# Faire une prédiction
prediction_normalized = model.predict(last_window)[0][0]
print(f"\nPrédiction normalisée: {prediction_normalized:.6f}")

# Dénormaliser pour avoir le prix réel en USD
prediction_real = prediction_normalized * scaler_std[0] + scaler_mean[0]
print(f"Prédiction prix BTC (USD): ${prediction_real:.2f}")