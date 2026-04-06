#!/usr/bin/env python3
"""
Module for concatenating multiple DataFrames.

This module provides utilities for combining multiple DataFrames with
custom indexing and hierarchical keys.
"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Concatenate two DataFrames with filtered data and hierarchical keys.

    Sets 'Timestamp' as the index for both DataFrames, filters df2 to
    include only rows with timestamps up to 1417411920, and concatenates
    the filtered df2 with df1 using hierarchical keys ('bitstamp' and
    'coinbase').

    Args:
        df1 (pd.DataFrame): 1st DataFrame (coinbase) with 'Timestamp' column.
        df2 (pd.DataFrame): 2nd DataFrame (bitstamp) with 'Timestamp' column.

    Returns:
        pd.DataFrame: A concatenated DataFrame with 'bitstamp' and 'coinbase'
                     as hierarchical keys, where df2 is filtered up to
                     timestamp 1417411920 and placed before df1 in the result.
    """
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:1417411920]
    return pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
