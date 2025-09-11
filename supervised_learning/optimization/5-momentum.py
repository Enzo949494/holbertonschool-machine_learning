#!/usr/bin/env python3
"""
Module for gradient descent with momentum optimization
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with
    momentum optimization algorithm

    Args:
        alpha: learning rate
        beta1: momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var

    Returns:
        the updated variable and the new moment, respectively
    """
    # Update momentum: v = beta1 * v + (1 - beta1) * grad
    v_new = beta1 * v + (1 - beta1) * grad

    # Update variable: var = var - alpha * v_new
    var_new = var - alpha * v_new

    return var_new, v_new
