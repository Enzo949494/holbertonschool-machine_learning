#!/usr/bin/env python3
"""
Module to transpose a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix

    Args: 2D matrix

    Returns: The transposed matrix.
    """
    rows = len(matrix)
    cols = len(matrix[0])

    transposed = []

    for c in range(cols):
        new_row = []
        for r in range(rows):
            new_row.append(matrix[r][c])
        transposed.append(new_row)

    return transposed
