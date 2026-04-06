#!/usr/bin/env python3

def slice(df):
    return df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]