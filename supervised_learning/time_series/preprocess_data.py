#!/usr/bin/env python3
"""
Version mémoire-optimisée: sauvegarde seulement les données normalisées.
Les fenêtres sont créées à la volée par tf.data.Dataset.map().
"""

import numpy as np
import pandas as pd
import tensorflow as tf

def load_and_merge(csv_paths):
    dfs = [pd.read_csv(path).dropna() for path in csv_paths]
    df_cb = dfs[0]
    df_bs = dfs[1]
    
    merged = pd.merge(df_cb, df_bs, on="Timestamp", suffixes=("_cb", "_bs"), how="inner")
    
    # Moyenne prix
    for col in ["Open", "High", "Low", "Close", "Weighted_Price"]:
        merged[col] = merged[f"{col}_cb"].fillna(merged[f"{col}_bs"])
    
    # Somme volumes
    merged["Volume_(BTC)"] = merged["Volume_(BTC)_cb"].fillna(0) + merged["Volume_(BTC)_bs"].fillna(0)
    merged["Volume_(Currency)"] = merged["Volume_(Currency)_cb"].fillna(0) + merged["Volume_(Currency)_bs"].fillna(0)
    
    merged = merged.sort_values("Timestamp").reset_index(drop=True)
    return merged.dropna(subset=["Close"])

def select_features(df):
    return df[["Close", "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price"]].values.astype(np.float32)

def main():
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
    print(f"Fichier: btc_normalized.npz ({values_normalized.nbytes/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
