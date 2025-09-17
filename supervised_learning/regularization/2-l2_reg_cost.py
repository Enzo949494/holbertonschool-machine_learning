#!/usr/bin/env python3
# filepath: /home/ko/holbertonschool-machine_learning/supervised_learning/regularization/2-l2_reg_cost.py
"""L2 Regularization Cost with TensorFlow"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: tensor containing the cost of the network without L2 regularization
        model: Keras model that includes layers with L2 regularization

    Returns:
        tensor containing the total cost for each layer of the network,
        accounting for L2 regularization
    """
    # Get all regularization losses from the model
    regularization_losses = model.losses
    
    # Create a list starting with the original cost
    costs = [cost]
    
    # Add each regularization loss to the list
    for reg_loss in regularization_losses:
        costs.append(reg_loss)
    
    # Stack all costs into a single tensor
    return tf.stack(costs)