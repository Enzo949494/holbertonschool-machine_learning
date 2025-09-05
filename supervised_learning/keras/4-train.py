#!/usr/bin/env python3
"""Module for training Keras models"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        verbose: boolean that determines if output should be printed during training
        shuffle: boolean that determines whether to shuffle the batches every epoch

    Returns:
        The History object generated after training the model
    """
    # Train the model using Keras fit method
    history = network.fit(
        data,                   # Input data (X_train)
        labels,                 # Target labels (Y_train_oh)
        batch_size=batch_size,  # Size of each batch for gradient descent
        epochs=epochs,          # Number of complete passes through the dataset
        verbose=verbose,        # Print training progress (True) or silent (False)
        shuffle=shuffle         # Shuffle data between epochs (False for reproducibility)
    )
    
    # Return the History object containing training metrics
    return history