#!/usr/bin/env python3
"""
Preprocess Bitcoin data from multiple exchanges.

This module loads and merges Bitcoin price data from Coinbase and Bitstamp
exchanges, normalizes the data using training set statistics, and saves
the processed data for model training.

The module uses memory-optimized approach by saving normalized data
and creating windows on-the-fly during training.

Functions:
    load_and_merge: Merge and clean data from multiple CSV files
    select_features: Extract relevant features from merged dataframe
    main: Execute the full preprocessing pipeline

Example:
    $ python3 preprocess_data.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf


def load_and_merge(csv_paths):
    """Load CSV files and merge Bitcoin data from multiple exchanges.

    Args:
        csv_paths (list): List of paths to CSV files to merge.
                         Expected format: [coinbase.csv, bitstamp.csv]

    Returns:
        pandas.DataFrame: Merged dataframe with columns for each feature,
                         sorted by Timestamp and with NaN values removed.

    Note:
        - Inner join is used to keep only common timestamps
        - Price columns are averaged between exchanges
        - Volume columns are summed across exchanges
    """
    dfs = [pd.read_csv(path).dropna() for path in csv_paths]
    df_cb = dfs[0]
    df_bs = dfs[1]

    merged = pd.merge(
        df_cb, df_bs, on="Timestamp", suffixes=("_cb", "_bs"), how="inner")

    # Moyenne prix
    for col in ["Open", "High", "Low", "Close", "Weighted_Price"]:
        merged[col] = merged[f"{col}_cb"].fillna(merged[f"{col}_bs"])

    # Somme volumes
    merged["Volume_(BTC)"] = merged
    ["Volume_(BTC)_cb"].fillna(0) + merged["Volume_(BTC)_bs"].fillna(0)

    merged["Volume_(Currency)"] = merged
    ["Volume_(Currency)_cb"].fillna(0) + merged
    ["Volume_(Currency)_bs"].fillna(0)

    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    return merged.dropna(subset=["Close"])


def select_features(df):
    """Extract relevant features from merged dataframe.

    Args:
        df (pandas.DataFrame): Merged Bitcoin dataframe with all columns.

    Returns:
        numpy.ndarray: Array of shape (n_samples, 4) with selected features:
                      - Close: Closing price
                      - Volume_(BTC): Volume in Bitcoin
                      - Volume_(Currency): Volume in currency
                      - Weighted_Price: Weighted average price
                      Values are cast to float32 for efficiency.
    """
    return df[["Close", "Volume_(BTC)", "Volume_(Currency)",
               "Weighted_Price"]].values.astype(np.float32)


def main():
    """Execute the complete preprocessing pipeline.

    This function:
    1. Loads and merges Bitcoin data from CSV files
    2. Selects relevant features
    3. Normalizes the data using training set statistics (70% split)
    4. Saves normalized data, original data, and scaler parameters to file

    Output:
        Saves btc_normalized.npz containing:
        - data: Normalized feature values (float32)
        - original_data: Original feature values (float32)
        - mean: Mean values used for normalization (float32)
        - std: Standard deviation values used for normalization (float32)
    """
    csv_paths = ["coinbase.csv", "bitstamp.csv"]

    print("🔄 Fusion...")
    df_merged = load_and_merge(csv_paths)
    values = select_features(df_merged)

    print(f"📊 {len(values)} lignes fusionnées")

    # Normalisation (fit sur 70% premiers)
    n = len(values)
    train_size = int(0.7 * n)

    train_data = values[:train_size]
    scaler_mean = np.mean(train_data, axis=0)
    scaler_std = np.std(train_data, axis=0)

    # Normalise TOUT le dataset
    values_normalized = (values - scaler_mean) / (scaler_std + 1e-6)

    # Sauvegarde les données normalisées ET originales
    np.savez("btc_normalized.npz",
             data=values_normalized.astype(np.float32),
             original_data=values.astype(np.float32),
             mean=scaler_mean.astype(np.float32),
             std=scaler_std.astype(np.float32))

    print("✅ Normalisation OK (light memory)")
    print(f"Fichier: btc_normalized.npz ({
        values_normalized.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
