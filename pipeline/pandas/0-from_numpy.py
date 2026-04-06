#!/usr/bin/env python3
"""
Module for converting numpy arrays to pandas DataFrames.

This module provides utilities for transforming numpy arrays into
pandas DataFrames with alphabetically labeled columns.
"""

import pandas as pd


def from_numpy(array):
    """
    Convert a numpy array to a pandas DataFrame.

    Creates a DataFrame from the input numpy array with columns
    labeled alphabetically (A, B, C, ...).

    Args:
        array: A 2D numpy array to be converted.

    Returns:
        A pandas DataFrame with the same data as the input array,
        with columns labeled A, B, C, etc. based on the number of columns.
    """
    n_cols = array.shape[1]
    columns = [chr(65 + i) for i in range(n_cols)]
    return pd.DataFrame(array, columns=columns)
