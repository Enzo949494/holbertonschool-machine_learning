#!/usr/bin/env python3
"""
Module for renaming and processing DataFrame columns.

This module provides utilities for renaming columns in DataFrames,
converting timestamps to datetime objects, and selecting specific columns.
"""

import pandas as pd


def rename(df):
    """
    Rename and process a DataFrame column.

    Renames the 'Timestamp' column to 'Datetime', converts the timestamp
    values from Unix time (seconds) to datetime objects, and selects only
    the 'Datetime' and 'Close' columns.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing 'Timestamp' and
                          'Close' columns.

    Returns:
        pd.DataFrame: A new DataFrame with 'Datetime' and 'Close' columns,
                     where 'Datetime' is converted to datetime format.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]
    return df
