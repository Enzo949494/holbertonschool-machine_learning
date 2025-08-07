#!/usr/bin/env python3
"""
Concatenate matrices specific axis using numpy
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis

    Args:mat1: first numpy array
         mat2: second numpy array
         axis: axis along to concatenate (default 0)

    Returns:new numpy.ndarray with concatenated matrices
    """
    return np.concatenate((mat1, mat2), axis=axis)
