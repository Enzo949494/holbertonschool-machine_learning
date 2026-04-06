#!/usr/bin/env python3
"""
Module for creating hierarchical DataFrames.

This module provides utilities for combining multiple DataFrames with
hierarchical multi-level indexing and sorting.
"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Create a hierarchical multi-indexed DataFrame from two DataFrames.

    Sets 'Timestamp' as the index for both DataFrames, filters them to
    the timestamp range [1417411980, 1417417980], concatenates them with
    hierarchical keys ('bitstamp' and 'coinbase'), swaps the index levels,
    and sorts by the first index level.

    Args:
        df1 (pd.DataFrame): 1st DataFrame (coinbase) with a 'Timestamp' column.
        df2 (pd.DataFrame): 2nd DataFrame (bitstamp) with a 'Timestamp' column.

    Returns:
        pd.DataFrame: A hierarchical DataFrame with multi-level index where the
                     1st level is the data source ('bitstamp', 'coinbase') and
                     the 2nd level is the timestamp, sorted by data source.
    """
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]
    df = pd.concat([df2, df1], keys=['bitstamp', 'coinbase'])
    df = df.swaplevel()
    df = df.sort_index(level=0)
    return df
