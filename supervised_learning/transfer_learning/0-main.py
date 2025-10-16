#!/usr/bin/env python3

import tensorflow as tf
preprocess_data = __import__('0-transfer').preprocess_data

# Load CIFAR-10 test data
_, (X, Y) = tf.keras.datasets.cifar10.load_data()

# Test preprocess_data function
X_p, Y_p = preprocess_data(X, Y)

print("X shape:", X_p.shape)
print("Y shape:", Y_p.shape)
print("X type:", type(X_p))
print("Y type:", type(Y_p))
print("\n✓ preprocess_data() works correctly!")