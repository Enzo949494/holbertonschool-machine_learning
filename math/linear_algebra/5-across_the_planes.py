#!/usr/bin/env python3
"""
Adds 2 2D matrix element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Adds 2 2D matrix element-wise

    Args: mat1 & 2 --> int/float

    Returns: sum of add
    """

    if len(mat1) != len(mat2):
        return None
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
    return [
        [a + b for a, b in zip(row1, row2)]
        for row1, row2 in zip(mat1, mat2)
    ]
