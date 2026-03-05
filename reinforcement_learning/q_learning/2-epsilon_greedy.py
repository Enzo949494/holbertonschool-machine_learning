#!/usr/bin/env python3

import numpy as np

def epsilon_greedy(Q, state, epsilon):
    p = np.random.uniform(0, 1)  # tire un nombre entre 0 et 1
    
    if p < epsilon:
        # EXPLORATION : action aléatoire
        return np.random.randint(0, Q.shape[1])
    else:
        # EXPLOITATION : meilleure action connue
        return np.argmax(Q[state])
