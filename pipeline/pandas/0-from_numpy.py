#!/usr/bin/env python3

import pandas as pd

def from_numpy(array):
    n_cols = array.shape[1]
    columns = [chr(65 + i) for i in range(n_cols)]
    return pd.DataFrame(array, columns=columns)
