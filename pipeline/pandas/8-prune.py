#!/usr/bin/env python3
"""
Module for cleaning DataFrames by removing missing values.

This module provides utilities for removing rows with missing data
in critical columns.
"""


def prune(df):
    """
    Remove rows with missing 'Close' values from the DataFrame.

    Drops all rows where the 'Close' column contains NaN (missing) values,
    returning a cleaned DataFrame with no missing data in that column.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing a 'Close' column
                          that may have missing values.

    Returns:
        pd.DataFrame: A DataFrame with rows containing NaN values in the
                     'Close' column removed.
    """
    return df.dropna(subset=['Close'])
