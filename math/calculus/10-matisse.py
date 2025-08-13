#!/usr/bin/env python3
"""
polynome
"""


def poly_derivative(poly):
    """
    python polynome
    """
    if (not isinstance(poly, list) or not poly or
            not all(isinstance(c, (int, float)) for c in poly)):
        return None
    if len(poly) == 1:
        return [0]
    deriv = [c * i for i, c in enumerate(poly)][1:]
    return deriv if any(deriv) else [0]
