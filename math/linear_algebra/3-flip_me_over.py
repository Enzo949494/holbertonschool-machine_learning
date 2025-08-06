#!/usr/bin/env python3
""""
transpose matrix
"""


import numpy as np


def matrix_transpose(matrix):
    """
    Transpose of a 2D matrix

    Args: matrix 2D

    Returns: transposed matrix
    """

    np_matrix = np.array(matrix)
    transposed = np_matrix.T
    return transposed.tolist()
