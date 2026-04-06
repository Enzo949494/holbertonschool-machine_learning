#!/usr/bin/env python3
"""
Module for slicing and downsampling DataFrame data.

This module provides utilities for selecting specific columns from DataFrames
and extracting rows at regular intervals for data downsampling.
"""


def slice(df):
    """
    Extract specific columns and downsample DataFrame by select every 60th row.

    Selects the columns 'High', 'Low', 'Close', and 'Volume_(BTC)' from the
    DataFrame and returns every 60th row, effectively downsampling the data
    by a factor of 60.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the columns 'High',
                          'Low', 'Close', and 'Volume_(BTC)'.

    Returns:
        pd.DataFrame: A DataFrame with the selected columns containing every
                     60th row from the input DataFrame.
    """
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]
