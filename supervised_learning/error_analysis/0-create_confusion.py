#!/usr/bin/env python3
"""
Module for creating confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes) containing
                the correct labels for each data point
        logits: one-hot numpy.ndarray of shape (m, classes) containing
                the predicted labels

    Returns:
        confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column indices representing
        the predicted labels
    """
    # Convert one-hot to class indices
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    # Get number of classes
    classes = labels.shape[1]

    # Initialize confusion matrix
    confusion = np.zeros((classes, classes))

    # Fill confusion matrix
    for i in range(len(true_labels)):
        true_class = true_labels[i]
        pred_class = predicted_labels[i]
        confusion[true_class, pred_class] += 1

    return confusion
