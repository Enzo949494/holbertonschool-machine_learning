#!/usr/bin/env python3
"""
Element-wise operations with numpy
"""


def np_elementwise(mat1, mat2):
    """
    Element-wise +, -, *, /
    
    Args:mat1: 1st numpy array
         mat2: 2nd numpy array or scalar
    
    Returns:tuple containing (sum, difference, product, quotient)
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
