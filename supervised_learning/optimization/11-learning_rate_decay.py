#!/usr/bin/env python3
"""
Module for learning rate decay using inverse time decay
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy
    
    Args:
        alpha: original learning rate
        decay_rate: weight used to determine the rate at which alpha will decay
        global_step: number of passes of gradient descent that have elapsed
        decay_step: number of passes of gradient descent that should occur before alpha is decayed further
    
    Returns:
        the updated value for alpha
    """
    # Calculate the number of decay periods that have elapsed
    decay_periods = global_step // decay_step
    
    # Apply inverse time decay formula in stepwise fashion
    updated_alpha = alpha / (1 + decay_rate * decay_periods)
    
    return updated_alpha