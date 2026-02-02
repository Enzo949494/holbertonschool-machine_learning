#!/usr/bin/env python3
"""
Evaluate the trained model and visualize predictions vs actual values

The model uses past 24 hours of BTC data to predict the next hour's price.
This script evaluates on the official test set (the future data).

Usage:
  python3 evaluate_model.py              # Official test set (default)
  python3 evaluate_model.py random       # Random period within test set
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load the trained LSTM model without compiling for faster loading
print("Loading model...")
model = keras.models.load_model('btc_model.h5', compile=False)
print("✅ Model loaded!")

# Load preprocessed data and normalization parameters
data = np.load('btc_normalized.npz')
values = data['data']  # Normalized features
original_values = data['original_data']  # Original Bitcoin prices in USD
scaler_mean = data['mean']  # Mean values from training set normalization
scaler_std = data['std']  # Std dev values from training set normalization

print(f"Loaded {len(values)} timesteps")

# Keep track of original indices before subsampling
original_len = len(values)
values = values[::20]
original_values = original_values[::20]  # Subsample original values too!
n = len(values)
print(f"Subsampled to {n} steps")

WINDOW_SIZE = 1440
HORIZON = 60


# Create windows
def make_windows(start, end):
    """Create training windows from time series data.

    Generates input-output pairs where input is a window of WINDOW_SIZE
    timesteps and output is the closing price WINDOW_SIZE + HORIZON ahead.

    Args:
        start (int): Starting index in the values array
        end (int): Ending index in the values array

    Returns:
        tuple: (X, y) where
            - X: numpy array of shape (n_windows, WINDOW_SIZE, 4) containing
                 input windows of normalized features
            - y: numpy array of shape (n_windows,) containing target normalized
                 closing prices
    """
    X, y = [], []
    for i in range(start, end - WINDOW_SIZE - HORIZON):
        X.append(values[i:i+WINDOW_SIZE])
        y.append(values[i+WINDOW_SIZE, 0])
    return np.array(X), np.array(y)


# Get test set (same split as training)
train_size = int(0.7 * n)
val_size = int(0.15 * n)
test_start = train_size + val_size
test_end = n

# Choose period based on command line argument
if len(sys.argv) > 1:
    period = sys.argv[1].lower()
    if period == "random":
        # Random window within test set,
        # but must leave room for WINDOW_SIZE + HORIZON
        max_start = test_end - WINDOW_SIZE - HORIZON
        if max_start > test_start:
            test_start = np.random.randint(test_start, max_start)
            print(f"📍 Period: RANDOM within test set (index {test_start})")
        else:
            print(f"📍 P: RANDOM unavailable (2 small). Use official test set.")
    else:
        print(f"⚠️  Unknown option '{period}'. Using official test set.")
else:
    # Default: official test set (70-100%)
    print("📍 Period: OFFICIAL TEST SET (70-100%, future data)")

test_X, test_y = make_windows(test_start, n)

# Get original prices for these test indices (REAL Bitcoin prices)
test_indices = np.arange(test_start, n - WINDOW_SIZE - HORIZON)
original_test_y = original_values[test_indices + WINDOW_SIZE, 0]

print(f"Test set: {len(test_X)} samples")

# Make predictions on test set using the trained model
print("\n🔮 Making predictions...")
predictions_norm = model.predict(test_X, verbose=0)

# Denormalize predictions to convert back to real USD prices
# Formula: real_price = normalized_price * (std + 1e-6) + mean
# Note: 1e-6 is added to prevent division by zero during normalization
predictions_usd = predictions_norm.flatten() * (
    scaler_std[0] + 1e-6) + scaler_mean[0]

print(f"✅ Predictions made!")
print(f"Mean Absolute Error: ${np.mean(np.abs(
    predictions_usd - original_test_y)):.2f}")
print(f"RMSE: ${np.sqrt(np.mean((predictions_usd - original_test_y)**2)):.2f}")

# Create visualization comparing actual vs predicted Bitcoin prices
plt.figure(figsize=(14, 6))
# Plot actual historical prices
plt.plot(
    original_test_y, label='Réel (Prix USD)', linewidth=2, color='#1f77b4')
# Plot model predictions
plt.plot(predictions_usd, label='Prédit (Prix USD)',
         linewidth=2, color='#ff7f0e', alpha=0.9)
# Configure plot labels and appearance
plt.title('Test Set: Actual vs Predicted BTC Price (USD)',
          fontsize=14, fontweight='bold')
plt.xlabel('Time Steps', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the comparison plot to file
plt.savefig('predictions_vs_actual.png', dpi=150, bbox_inches='tight')
print("\n📊 Plot saved as 'predictions_vs_actual.png'")
plt.show()
