#!/usr/bin/env python3
"""
arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Adds 2 arrays element-wise

    Args: arr 1 & 2 --> int/float

    Returns: eleement-wise sum
    """

    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
