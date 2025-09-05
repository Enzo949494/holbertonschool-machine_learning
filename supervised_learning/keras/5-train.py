#!/usr/bin/env python3
"""Module for training Keras models"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) contain the labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        verbose: bool that determines if output should be printed during train
        shuffle: bool that determines whether to shuffle the batche every epoch

    Returns:
        The History object generated after training the model
    """
    # Train the model using Keras fit method
    history = network.fit(
        data,                   		# Input data (X_train)
        labels,                 		# Target labels (Y_train_oh)
        batch_size=batch_size,  		# Size of each batch for gradient descent
        epochs=epochs,          		# Number of complete passes through the dataset
        validation_data=validation_data, # valid data for monit
		verbose=verbose,        		# Print train progress (True) or silent (False)
        shuffle=shuffle         		# Shuffle data btwn epoch (Fals for reproduc)
    )

    # Return the History object containing training metrics
    return history
