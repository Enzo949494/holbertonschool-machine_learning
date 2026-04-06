#!/usr/bin/env python3

import pandas as pd


def array(df):
    return df[['High', 'Close']].tail(10).to_numpy()
