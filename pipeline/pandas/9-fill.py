#!/usr/bin/env python3
"""
Module for handling missing values in DataFrames.

This module provides utilities for cleaning DataFrames by removing
unnecessary columns and filling missing values with appropriate strategies.
"""


def fill(df):
    """
    Remove unnecessary columns and fill missing values in the DataFrame.

    Drops the 'Weighted_Price' column, forward-fills the 'Close' column,
    and fills missing values in 'High', 'Low', and 'Open' columns with the
    'Close' value. Volume columns are filled with 0.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing price and volume
                          columns including 'Close', 'High', 'Low', 'Open',
                          'Volume_(BTC)', 'Volume_(Currency)', and optionally
                          'Weighted_Price'.

    Returns:
        pd.DataFrame: A cleaned DataFrame with 'Weighted_Price' removed,
                     missing values filled appropriately, and volume columns
                     with 0 for missing values.
    """
    df = df.drop(columns=['Weighted_Price'])
    df['Close'] = df['Close'].ffill()
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
