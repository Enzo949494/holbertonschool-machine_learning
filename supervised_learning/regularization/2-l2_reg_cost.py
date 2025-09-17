#!/usr/bin/env python3
"""Module for computing neural network L2 regularization cost."""

import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates a tensor where each element is the sum of the initial cost without regularization
    and the L2 regularization penalty for a layer in the model.

    Args:
        cost: tensor of the cost without L2 regularization
        model: Keras model with layers using L2 regularization

    Returns:
        Tensor of size (number of layers with L2) containing:
        [cost + regularization layer 1, cost + regularization layer 2, ...]
    """
    costs_per_layer = []

    for layer in model.layers:
        # Check if the layer has losses (regularization losses, typically L2)
        if hasattr(layer, 'losses') and layer.losses:
            # Sum of L2 penalties for this layer
            layer_l2_penalty = tf.add_n(layer.losses)
            # Add total cost for this layer (cost without regularization + L2 penalty)
            costs_per_layer.append(cost + layer_l2_penalty)

    # Convert the list into a float32 tensor for TensorFlow compatibility
    return tf.convert_to_tensor(costs_per_layer, dtype=tf.float32)
