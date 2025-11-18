#!/usr/bin/env python3
"""
Module for calculating matrix cofactors.
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix.

    Args:
        matrix: A list of lists whose cofactor matrix should be calculated

    Returns:
        The cofactor matrix of matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square or is empty
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # Check if matrix is square
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # Import minor function
    minor_func = __import__('1-minor').minor

    # Get the minor matrix
    minor_matrix = minor_func(matrix)

    # Calculate cofactor matrix: cofactor = (-1)^(i+j) * minor
    cofactor_matrix = []
    for i in range(n):
        cofactor_row = []
        for j in range(n):
            cofactor_row.append(((-1) ** (i + j)) * minor_matrix[i][j])
        cofactor_matrix.append(cofactor_row)

    return cofactor_matrix
