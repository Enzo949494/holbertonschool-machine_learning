#!/usr/bin/env python3
"""
Module for calculating matrix adjugates.
"""


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix.

    Args:
        matrix: A list of lists whose adjugate matrix should be calculated

    Returns:
        The adjugate matrix of matrix

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

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # Import cofactor function
    cofactor_func = __import__('2-cofactor').cofactor

    # Get the cofactor matrix
    cofactor_matrix = cofactor_func(matrix)

    # Adjugate is the transpose of the cofactor matrix
    adjugate_matrix = []
    for i in range(n):
        adjugate_row = []
        for j in range(n):
            adjugate_row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(adjugate_row)

    return adjugate_matrix
