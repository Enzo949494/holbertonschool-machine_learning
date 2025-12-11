#!/usr/bin/env python3
"""Bayesian Optimization of a Neural Network using GPyOpt"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import GPyOpt
from GPyOpt.methods import BayesianOptimization
import json
from datetime import datetime

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convert to categorical
y_train_cat = keras.utils.to_categorical(y_train, 3)
y_test_cat = keras.utils.to_categorical(y_test, 3)

# Store for convergence tracking
best_loss = float('inf')
iteration_count = 0
losses = []
iterations = []


def build_and_train_model(params):
    """
    Build and train a neural network with given hyperparameters
    
    Args:
        params: array of hyperparameters [learning_rate, units1, dropout, l2_reg, batch_size]
    
    Returns:
        Validation loss (negative for maximization in GPyOpt)
    """
    global best_loss, iteration_count, losses, iterations
    
    learning_rate = float(params[0][0])
    units1 = int(params[0][1])
    dropout = float(params[0][2])
    l2_reg = float(params[0][3])
    batch_size = int(params[0][4])
    
    iteration_count += 1
    
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
        )
        
        # Train
        history = model.fit(
            X_train, y_train_cat,
            validation_split=0.2,
            epochs=100,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        val_loss = min(history.history['val_loss'])
        
        # Save checkpoint if best so far
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_name = (
                f"best_model_lr{learning_rate:.4f}_u{units1}_"
                f"d{dropout:.2f}_l2{l2_reg:.4f}_bs{batch_size}.h5"
            )
            model.save(checkpoint_name)
        
        # Track convergence
        losses.append(val_loss)
        iterations.append(iteration_count)
        
        print(f"Iteration {iteration_count}: Loss={val_loss:.4f}, "
              f"LR={learning_rate:.4f}, Units={units1}, "
              f"Dropout={dropout:.2f}, L2={l2_reg:.4f}, BS={batch_size}")
        
        return val_loss
    
    except Exception as e:
        print(f"Error in iteration {iteration_count}: {e}")
        return 1000.0  # Return high loss on error


# Define bounds for hyperparameters
bounds = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
    {'name': 'units', 'type': 'discrete', 'domain': (16, 32, 64, 128)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.0, 0.5)},
    {'name': 'l2_regularization', 'type': 'continuous', 'domain': (0.0, 0.01)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (8, 16, 32, 64)}
]

# Run Bayesian Optimization
optimizer = BayesianOptimization(
    f=build_and_train_model,
    domain=bounds,
    num_cores=1,
    initial_design_numdata=5,
    acquisition_type='EI',
    exact_feval=True,
)

optimizer.run_optimization(max_iter=25, verbosity=False)

# Get best parameters
best_params = optimizer.x_opt
best_loss_opt = optimizer.fx_opt

print("\n" + "="*50)
print("OPTIMIZATION COMPLETE")
print("="*50)
print(f"Best Loss: {best_loss_opt:.4f}")
print(f"Best Learning Rate: {best_params[0]:.4f}")
print(f"Best Units: {int(best_params[1])}")
print(f"Best Dropout: {best_params[2]:.4f}")
print(f"Best L2 Regularization: {best_params[3]:.4f}")
print(f"Best Batch Size: {int(best_params[4])}")

# Generate report
report = {
    'timestamp': datetime.now().isoformat(),
    'total_iterations': iteration_count,
    'best_loss': float(best_loss_opt),
    'best_parameters': {
        'learning_rate': float(best_params[0]),
        'units': int(best_params[1]),
        'dropout': float(best_params[2]),
        'l2_regularization': float(best_params[3]),
        'batch_size': int(best_params[4])
    },
    'optimization_history': {
        'iterations': iterations,
        'losses': [float(l) for l in losses]
    }
}

# Save report
with open('bayes_opt.txt', 'w') as f:
    f.write("BAYESIAN OPTIMIZATION REPORT\n")
    f.write("="*50 + "\n")
    f.write(f"Timestamp: {report['timestamp']}\n")
    f.write(f"Total Iterations: {report['total_iterations']}\n")
    f.write(f"Best Validation Loss: {report['best_loss']:.4f}\n\n")
    f.write("BEST HYPERPARAMETERS:\n")
    f.write("-"*50 + "\n")
    f.write(f"Learning Rate: {report['best_parameters']['learning_rate']:.4f}\n")
    f.write(f"Units (Layer 1): {report['best_parameters']['units']}\n")
    f.write(f"Dropout Rate: {report['best_parameters']['dropout']:.4f}\n")
    f.write(f"L2 Regularization: {report['best_parameters']['l2_regularization']:.4f}\n")
    f.write(f"Batch Size: {report['best_parameters']['batch_size']}\n")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(iterations, losses, 'b-o', linewidth=2, markersize=6)
plt.axhline(y=best_loss_opt, color='r', linestyle='--', label=f'Best Loss: {best_loss_opt:.4f}')
plt.xlabel('Iteration')
plt.ylabel('Validation Loss')
plt.title('Bayesian Optimization Convergence')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('convergence.png', dpi=100)
plt.show()

print(f"\nReport saved to 'bayes_opt.txt'")
print(f"Convergence plot saved to 'convergence.png'")