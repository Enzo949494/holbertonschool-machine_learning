#!/usr/bin/env python3
"""
Module for calculating sensitivity from confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
                  where row indices represent the correct labels and
                  column indices represent the predicted labels

    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity
        of each class
    """
    # Get the number of classes
    classes = confusion.shape[0]

    # Initialize sensitivity array
    sensitivity_array = np.zeros(classes)

    # Calculate sensitivity for each class
    for i in range(classes):
        # True positives: diagonal element for class i
        true_positives = confusion[i, i]

        # Total actual positives: sum of row i (all actual class i samples)
        total_actual_positives = np.sum(confusion[i, :])

        # Sensitivity = TP / (TP + FN) = TP / Total_Actual_Positives
        if total_actual_positives > 0:
            sensitivity_array[i] = true_positives / total_actual_positives
        else:
            sensitivity_array[i] = 0

    return sensitivity_array
