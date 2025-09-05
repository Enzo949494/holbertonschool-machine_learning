#!/usr/bin/env python3
"""Module for training Keras models"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing
                labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient
                descent
        validation_data: data to validate the model with, if not None
        early_stopping: boolean that indicates whether early stopping
                       should be used
        patience: patience used for early stopping
        learning_rate_decay: boolean that indicates whether learning rate
                            decay should be used
        alpha: initial learning rate
        decay_rate: decay rate
        save_best: boolean indicating whether to save the model after each
                  epoch if it is the best
        filepath: file path where the model should be saved
        verbose: boolean that determines if output should be printed
                during training
        shuffle: boolean that determines whether to shuffle the batches
                every epoch

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

    # Add model checkpoint if conditions are met
    if save_best and filepath is not None:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,      # Path where to save the model
            monitor='val_loss',     # Monitor validation loss
            save_best_only=True,    # Save only the best model
            mode='min',             # Minimize validation loss
            verbose=0               # Silent saving
        )
        callbacks.append(checkpoint)

    # Train the model using Keras fit method
    history = network.fit(
        data,                           # Input data (X_train)
        labels,                         # Target labels (Y_train_oh)
        batch_size=batch_size,          # Size of each batch
        epochs=epochs,                  # Number of complete passes
        validation_data=validation_data,  # Validation data for monitoring
        callbacks=callbacks,            # List of callbacks
        verbose=verbose,                # Print training progress
        shuffle=shuffle                 # Shuffle data between epochs
    )

    # Return the History object containing training metrics
    return history
