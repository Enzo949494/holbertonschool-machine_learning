#!/usr/bin/env python3
"""Module for optimizing Keras models"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimization for a keras model with categorical crossentropy 
    loss and accuracy metrics

    Args:
        network: the model to optimize
        alpha: the learning rate
        beta1: the first Adam optimization parameter
        beta2: the second Adam optimization parameter

    Returns:
        None
    """
    # Create Adam optimizer with specified parameters
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,    # Learning rate (step size)
        beta_1=beta1,          # First moment decay rate
        beta_2=beta2           # Second moment decay rate
    )
    
    # Compile the model with Adam optimizer, categorical crossentropy loss,
    # and accuracy metrics
    network.compile(
        optimizer=optimizer,                    # Adam optimizer with custom parameters
        loss='categorical_crossentropy',        # Loss function for multi-class classification
        metrics=['accuracy']                    # Track accuracy during training
    )
    
    # Add compatibility for older Keras versions (add lr attribute)
    if not hasattr(optimizer, 'lr'):
        optimizer.lr = optimizer.learning_rate