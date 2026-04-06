#!/usr/bin/env python3
"""
Module for sorting and transposing DataFrames.

This module provides utilities for reordering DataFrame data by sorting
and transposing rows and columns.
"""


def flip_switch(df):
    """
    Sort DataFrame by 'Timestamp' in descending order and transpose it.

    Sorts the DataFrame by the 'Timestamp' column in descending order
    (newest first), then transposes the DataFrame so that rows become
    columns and columns become rows.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: A transposed DataFrame sorted by 'Timestamp' in
                     descending order, where original columns become rows
                     and original rows become columns.
    """
    return df.sort_values(by='Timestamp', ascending=False).T
