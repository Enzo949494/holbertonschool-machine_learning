#!/usr/bin/env python3

import numpy as np

def positional_encoding(max_seq_len, dm):
    # Positions: (max_seq_len, 1)
    position = np.arange(max_seq_len)[:, np.newaxis]
    
    # Dimensions: (1, dm)
    div_term = np.exp(np.arange(0, dm, 2) * -(np.log(10000.0) / dm))
    
    # Angles: (max_seq_len, dm/2)
    pe = np.zeros((max_seq_len, dm))
    pe[:, 0:dm:2] = np.sin(position * div_term)  # sin sur pairs
    pe[:, 1:dm:2] = np.cos(position * div_term)  # cos sur impairs
    
    return pe
