#!/usr/bin/env python3
""""
calculate shape of a matrix
"""


def matrix_shape(matrix):
    """
    Calculate shape of a matrix

    Args: matrix list: list representing a matrix

    Returns: list of integers who represente shape of the matrix
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) == 0:
            break
        matrix = matrix[0]
    return shape
