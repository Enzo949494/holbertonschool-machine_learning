#!/usr/bin/env python3
"""
Module for calculating specificity from confusion matrix
"""

import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
    
    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes) 
                  where row indices represent the correct labels and 
                  column indices represent the predicted labels
    
    Returns:
        numpy.ndarray of shape (classes,) containing the specificity 
        of each class
    """
    # Get the number of classes
    classes = confusion.shape[0]
    
    # Initialize specificity array
    specificity_array = np.zeros(classes)
    
    # Calculate specificity for each class
    for i in range(classes):
        # True positives: diagonal element for class i
        true_positives = confusion[i, i]
        
        # False positives: sum of column i minus true positives
        false_positives = np.sum(confusion[:, i]) - true_positives
        
        # False negatives: sum of row i minus true positives
        false_negatives = np.sum(confusion[i, :]) - true_positives
        
        # True negatives: total minus TP, FP, FN
        total_samples = np.sum(confusion)
        true_negatives = total_samples - true_positives - false_positives - false_negatives
        
        # Specificity = TN / (TN + FP)
        if (true_negatives + false_positives) > 0:
            specificity_array[i] = true_negatives / (true_negatives + false_positives)
        else:
            specificity_array[i] = 0
    
    return specificity_array