#!/usr/bin/env python3
"""
Module for setting DataFrame indices.

This module provides utilities for setting a column as the DataFrame index
to enable advanced indexing and time-series operations.
"""


def index(df):
    """
    Set the 'Timestamp' column as the DataFrame index.

    Converts the 'Timestamp' column into the DataFrame's index, which is
    useful for time-series data operations and enables index-based access.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: A DataFrame with 'Timestamp' set as the index instead
                     of a regular column.
    """
    return df.set_index('Timestamp')
