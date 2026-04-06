#!/usr/bin/env python3
"""
Module for extracting array data from DataFrames.

This module provides utilities for converting DataFrame data into
numpy arrays with specific column selections.
"""


def array(df):
    """
    Extract the last 10 rows of 'High' and 'Close' columns as a numpy array.

    Selects the 'High' and 'Close' columns from the DataFrame, retrieves
    the last 10 rows, and converts them to a numpy array.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing 'High' and 'Close'
                          columns.

    Returns:
        np.ndarray: A 2D numpy array of shape (10, 2) containing the last
                   10 rows of 'High' and 'Close' columns.
    """
    return df[['High', 'Close']].tail(10).to_numpy()
