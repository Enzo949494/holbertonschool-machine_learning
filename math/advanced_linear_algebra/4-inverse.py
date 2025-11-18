#!/usr/bin/env python3
"""
Module for calculating matrix inverses.
"""


def inverse(matrix):
    """
    Calculates the inverse of a matrix.

    Args:
        matrix: A list of lists whose inverse should be calculated

    Returns:
        The inverse of matrix, or None if matrix is singular

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

    # Import determinant and adjugate functions
    determinant_func = __import__('0-determinant').determinant
    adjugate_func = __import__('3-adjugate').adjugate

    # Calculate determinant
    det = determinant_func(matrix)

    # If determinant is 0, matrix is singular
    if det == 0:
        return None

    # Get adjugate matrix
    adj_matrix = adjugate_func(matrix)

    # Inverse = (1/det) * adjugate
    inverse_matrix = []
    for i in range(n):
        inverse_row = []
        for j in range(n):
            inverse_row.append(adj_matrix[i][j] / det)
        inverse_matrix.append(inverse_row)

    return inverse_matrix
