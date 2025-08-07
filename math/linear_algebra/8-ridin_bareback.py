#!/usr/bin/env python3
"""
Multiply two 2D matrix
"""


def mat_mul(mat1, mat2):
    """
    Multiplies 2 2D matrix mat1 & 2

    Args:mat1: 1st matrix (m x n)
         mat2: 2nd matrix (p x q)

    Returns: Resulting matrix (m x q) if multiplication possible, else None
    """
    if not mat1 or not mat2:
        return None
    if any(not isinstance(row, list) for row in mat1 + mat2):
        return None

    m, n = len(mat1), len(mat1[0])
    p, q = len(mat2), len(mat2[0])

    if (any(len(row) != n for row in mat1) or
            any(len(row) != q for row in mat2)):
        return None

    if n != p:
        return None

    return [
        [
            sum(mat1[i][k] * mat2[k][j] for k in range(n))
            for j in range(q)
        ]
        for i in range(m)
    ]
