#!/usr/bin/env python3
"""
sigma fonction
"""


def summation_i_squared(n):
    """
    code for sigma
    """
    if not isinstance(n, int) or n < 1:
        return None
    return sum(i**2 for i in range(1, n+1))
