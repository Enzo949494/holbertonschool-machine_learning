#!/usr/bin/env python3
"""
Module for calculating precision from confusion matrix
"""

import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix
    
    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes) 
                  where row indices represent the correct labels and 
                  column indices represent the predicted labels
    
    Returns:
        numpy.ndarray of shape (classes,) containing the precision 
        of each class
    """
    # Get the number of classes
    classes = confusion.shape[0]
    
    # Initialize precision array
    precision_array = np.zeros(classes)
    
    # Calculate precision for each class
    for i in range(classes):
        # True positives: diagonal element for class i
        true_positives = confusion[i, i]
        
        # Total predicted positives: sum of column i (all predicted class i samples)
        total_predicted_positives = np.sum(confusion[:, i])
        
        # Precision = TP / (TP + FP) = TP / Total_Predicted_Positives
        if total_predicted_positives > 0:
            precision_array[i] = true_positives / total_predicted_positives
        else:
            precision_array[i] = 0
    
    return precision_array