#!/usr/bin/env python3
"""
Module for loading data from CSV files into pandas DataFrames.

This module provides utilities for reading CSV files with custom delimiters
and converting them into pandas DataFrames.
"""

import pandas as pd


def from_file(filename, delimiter):
    """
    Load a CSV file into a pandas DataFrame.

    Reads a CSV file with a specified delimiter and returns it as a
    pandas DataFrame.

    Args:
        filename (str): Path to the CSV file to read.
        delimiter (str): The delimiter character used in the CSV file
                        (e.g., ',', ';', '\t').

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(filename, sep=delimiter)
