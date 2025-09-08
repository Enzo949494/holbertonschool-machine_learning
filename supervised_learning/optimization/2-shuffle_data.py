#!/usr/bin/env python3
"""
Module for shuffling data points in matrices
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    """
    # Créer une liste d'indices mélangés
    permutation = np.random.permutation(X.shape[0])
    
    # Réorganiser X et Y avec les mêmes indices
    return X[permutation], Y[permutation]