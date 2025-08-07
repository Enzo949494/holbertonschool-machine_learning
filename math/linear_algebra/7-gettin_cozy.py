#!/usr/bin/env python3
"""
Concatenate 2 2D matrix along a given axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate 2 2D matrix (lists of lists) along axis 0 or 1.

    Args:mat1 & 2: 2D matrix (ints/floats)
         axis (int): 0 for vertical concat (add rows)
                     1 for horizontal concat (add columns)

    Returns: concatenated matrix if possible, else None
    """
    if axis not in (0, 1):
        return None

    if not (mat1 and mat2):
        return None
    if not (all(isinstance(row, list) for row in mat1) and
            all(isinstance(row, list) for row in mat2)):
        return None

    if axis == 0:
        ncols1 = len(mat1[0])
        ncols2 = len(mat2[0])
        if (any(len(row) != ncols1 for row in mat1) or
                any(len(row) != ncols2 for row in mat2)):
            return None
        if ncols1 != ncols2:
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    else:
        if len(mat1) != len(mat2):
            return None
        ncols1 = len(mat1[0])
        ncols2 = len(mat2[0])
        if (any(len(row) != ncols1 for row in mat1) or
                any(len(row) != ncols2 for row in mat2)):
            return None
        return [row1[:] + row2[:] for row1, row2 in zip(mat1, mat2)]
