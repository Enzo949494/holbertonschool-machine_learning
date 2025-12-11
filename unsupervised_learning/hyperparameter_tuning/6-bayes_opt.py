#!/usr/bin/env python3
"""Bayesian Optimization of a Neural Network using GPyOpt"""

import os
import random
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers

from GPyOpt.methods import BayesianOptimization


# ======================================================================
# Fixer les seeds pour rendre le script (relativement) déterministe
# ======================================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)   # fixe tf.random + keras backend [web:89]


# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=SEED
)

# Convert to categorical
y_train_cat = keras.utils.to_categorical(y_train, 3)
y_test_cat = keras.utils.to_categorical(y_test, 3)

# Global for best checkpoint & iteration counter
best_loss = float('inf')
iteration_counter = 0


def build_and_train_model(params):
    """
    Build and train a neural network with given hyperparameters.

    Args:
        params: array of hyperparameters
                [learning_rate, units1, dropout, l2_reg, batch_size]

    Returns:
        Validation loss (to be minimized by GPyOpt)
    """
    global best_loss, iteration_counter

    iteration_counter += 1

    learning_rate = float(params[0][0])
    units1 = int(params[0][1])
    dropout = float(params[0][2])
    l2_reg = float(params[0][3])
    batch_size = int(params[0][4])

    try:
        # Build model
        model = keras.Sequential([
            layers.Dense(
                units1,
                activation='relu',
                input_shape=(4,),
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ),
            layers.Dropout(dropout),
            layers.Dense(
                units1 // 2,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(l2_reg)
            ),
            layers.Dropout(dropout),
            layers.Dense(3, activation='softmax')
        ])

        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )  # [web:97]

        # Train
        history = model.fit(
            X_train, y_train_cat,
            validation_split=0.2,
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )

        # Best val_loss during training
        val_loss = float(min(history.history['val_loss']))

        # Save checkpoint if best so far
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_name = (
                f"best_model_lr{learning_rate:.4f}_u{units1}_"
                f"d{dropout:.2f}_l2{l2_reg:.4f}_bs{batch_size}.h5"
            )
            model.save(checkpoint_name)  # [web:93][web:96]

        # Log clair par itération
        print(
            f"Iteration {iteration_counter:02d} | "
            f"Loss={val_loss:.4f}, "
            f"LR={learning_rate:.4f}, Units={units1}, "
            f"Dropout={dropout:.2f}, L2={l2_reg:.4f}, BS={batch_size}"
        )

        return val_loss

    except Exception as e:
        print(f"Iteration {iteration_counter:02d} | Error during evaluation: {e}")
        # Grande loss pour que BO évite cette zone
        return 1e3


# Define bounds for hyperparameters (5 hyperparams) [web:72]
bounds = [
    {'name': 'learning_rate',     'type': 'continuous', 'domain': (0.0001, 0.1)},
    {'name': 'units',             'type': 'discrete',   'domain': (16, 32, 64, 128)},
    {'name': 'dropout',           'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_regularization', 'type': 'continuous', 'domain': (0.0, 0.01)},
    {'name': 'batch_size',        'type': 'discrete',   'domain': (8, 16, 32, 64)}
]


# Run Bayesian Optimization (max 30 iterations) [web:76]
optimizer = BayesianOptimization(
    f=build_and_train_model,
    domain=bounds,
    initial_design_numdata=5,
    acquisition_type='EI',
    exact_feval=True
)

optimizer.run_optimization(max_iter=25, verbosity=True)

# Best parameters from GPyOpt [web:73]
best_params = optimizer.x_opt
best_loss_opt = float(optimizer.fx_opt)

print("\n" + "=" * 50)
print("OPTIMIZATION COMPLETE")
print("=" * 50)
print(f"Best Loss: {best_loss_opt:.4f}")
print(f"Best Learning Rate: {best_params[0]:.4f}")
print(f"Best Units: {int(best_params[1])}")
print(f"Best Dropout: {best_params[2]:.4f}")
print(f"Best L2 Regularization: {best_params[3]:.4f}")
print(f"Best Batch Size: {int(best_params[4])}")

# Save official GPyOpt report [web:73][web:103]
optimizer.save_report('bayes_opt.txt')

optimizer.plot_convergence(filename='convergence.png')

print("\nReport saved to 'bayes_opt.txt'")
print("Convergence plot saved to 'convergence.png'")
