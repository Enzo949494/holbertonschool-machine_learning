#!/usr/bin/env python3
"""
Module for calculating matrix determinants.
"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix.

    Args:
        matrix: A list of lists whose determinant should be calculated

    Returns:
        The determinant of the matrix

    Raises:
        TypeError: If matrix is not a list of lists
        ValueError: If matrix is not square
    """
    # Check if matrix is a list of lists
    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # Handle 0x0 matrix
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # Check if matrix is square
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")

    # Base cases
    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive cofactor expansion
    det = 0
    for j in range(n):
        # Create minor by removing row 0 and column j
        minor = [row[:j] + row[j+1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(minor)

    return det
