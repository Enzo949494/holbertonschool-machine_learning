#!/usr/bin/env python3
"""
Module for analyzing DataFrame statistics.

This module provides utilities for generating descriptive statistics
from DataFrames containing numerical data.
"""


def analyze(df):
    """
    Generate descriptive statistics for numerical columns in a DataFrame.

    Removes the 'Timestamp' column and returns statistical summaries
    (count, mean, std, min, 25%, 50%, 75%, max) for all remaining
    numerical columns.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing numerical data
                          and a 'Timestamp' column.

    Returns:
        pd.DataFrame: A DataFrame containing descriptive statistics for
                     each numerical column, with statistics as rows
                     (count, mean, std, min, 25%, 50%, 75%, max).
    """
    return df.drop(columns=['Timestamp']).describe()
