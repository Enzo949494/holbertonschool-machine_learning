#!/usr/bin/env python3
"""
Module for sorting DataFrames by price values.

This module provides utilities for sorting DataFrames by the 'High' column
to find maximum price values.
"""


def high(df):
    """
    Sort DataFrame by 'High' column in descending order.

    Sorts the DataFrame by the 'High' column in descending order so that
    rows with the highest prices appear first.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing a 'High' column
                          with price values.

    Returns:
        pd.DataFrame: A DataFrame sorted by 'High' values in descending order,
                     with the highest prices at the top.
    """
    return df.sort_values(by='High', ascending=False)
