#!/usr/bin/env python3
"""
Evaluate the trained model and visualize predictions vs actual values
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load model
print("Loading model...")
model = keras.models.load_model('btc_model.h5', compile=False)
print("✅ Model loaded!")

# Load data
data = np.load('btc_normalized.npz')
values = data['data']
scaler_mean = data['mean']
scaler_std = data['std']

print(f"Loaded {len(values)} timesteps")

# Subsample (same as training)
values = values[::20]
n = len(values)
print(f"Subsampled to {n} steps")

WINDOW_SIZE = 1440
HORIZON = 60

# Create windows
def make_windows(start, end):
    X, y = [], []
    for i in range(start, end - WINDOW_SIZE - HORIZON):
        X.append(values[i:i+WINDOW_SIZE])
        y.append(values[i+WINDOW_SIZE, 0])
    return np.array(X), np.array(y)

# Get test set (same split as training)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
test_start = train_size + val_size

test_X, test_y = make_windows(test_start, n)

print(f"Test set: {len(test_X)} samples")

# Make predictions
print("\n🔮 Making predictions...")
predictions_norm = model.predict(test_X, verbose=0)

# Denormalize
predictions = predictions_norm * scaler_std[0] + scaler_mean[0]
actual = test_y * scaler_std[0] + scaler_mean[0]

print(f"✅ Predictions made!")
print(f"Mean Absolute Error: ${np.mean(np.abs(predictions - actual)):.2f}")
print(f"RMSE: ${np.sqrt(np.mean((predictions - actual)**2)):.2f}")

# Plot
plt.figure(figsize=(14, 6))
plt.plot(actual, label='Réel', linewidth=2, color='#1f77b4')
plt.plot(predictions, label='Prédit', linewidth=2, color='#ff7f0e', alpha=0.9)
plt.title('Test Set: Actual vs Predicted BTC Price', fontsize=14, fontweight='bold')
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Price (Normalized Units)', fontsize=12)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig('predictions_vs_actual.png', dpi=150, bbox_inches='tight')
print("\n📊 Plot saved as 'predictions_vs_actual.png'")
plt.show()
