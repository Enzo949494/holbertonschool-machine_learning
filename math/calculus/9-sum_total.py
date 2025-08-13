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
    return n * (n + 1) * (2 * n + 1) // 6

