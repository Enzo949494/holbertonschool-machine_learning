#!/usr/bin/env python3
"""Module for training Keras models"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        validation_data: data to validate the model with, if not None
        early_stopping: boolean that indicates whether early stopping should be used
        patience: patience used for early stopping
        learning_rate_decay: boolean that indicates whether learning rate decay should be used
        alpha: initial learning rate
        decay_rate: decay rate
        verbose: boolean that determines if output should be printed during training
        shuffle: boolean that determines whether to shuffle the batches every epoch

    Returns:
        The History object generated after training the model
    """
    # Initialize callbacks list
    callbacks = []
    
    # Add early stopping if conditions are met
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',     # Monitor validation loss
            patience=patience,      # Number of epochs with no improvement
            restore_best_weights=True  # Restore best weights when stopping
        )
        callbacks.append(early_stop)
    
    # Add learning rate decay if conditions are met
    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            """
            Inverse time decay function
            lr = alpha / (1 + decay_rate * epoch)
            """
            return alpha / (1 + decay_rate * epoch)
        
        lr_scheduler = K.callbacks.LearningRateScheduler(
            scheduler,      # Function that calculates new learning rate
            verbose=1       # Print learning rate updates
        )
        callbacks.append(lr_scheduler)
    
    # Train the model using Keras fit method
    history = network.fit(
        data,                           # Input data (X_train)
        labels,                         # Target labels (Y_train_oh)
        batch_size=batch_size,          # Size of each batch for gradient descent
        epochs=epochs,                  # Number of complete passes through dataset
        validation_data=validation_data, # Validation data for monitoring
        callbacks=callbacks,            # List of callbacks (including early stopping and lr decay)
        verbose=verbose,                # Print training progress
        shuffle=shuffle                 # Shuffle data between epochs
    )
    
    # Return the History object containing training metrics
    return history
