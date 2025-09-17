#!/usr/bin/env python3
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
    reg_losses = model.losses[:2]  # les 2 couches cachées avec L2 seulement
    total_reg_loss = tf.reduce_sum(reg_losses)
    total_cost = cost + total_reg_loss
    return tf.stack([total_cost] + reg_losses)
