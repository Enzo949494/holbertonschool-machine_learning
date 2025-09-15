#!/usr/bin/env python3
"""
Module for calculating F1 score from confusion matrix
"""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix
    
    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes) 
                  where row indices represent the correct labels and 
                  column indices represent the predicted labels
    
    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score 
        of each class
    """
    # Get sensitivity (recall) and precision for each class
    sens = sensitivity(confusion)
    prec = precision(confusion)
    
    # Calculate F1 score using harmonic mean formula
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.zeros(sens.shape)
    
    # Avoid division by zero
    for i in range(len(sens)):
        if (prec[i] + sens[i]) > 0:
            f1[i] = 2 * (prec[i] * sens[i]) / (prec[i] + sens[i])
        else:
            f1[i] = 0
    
    return f1
