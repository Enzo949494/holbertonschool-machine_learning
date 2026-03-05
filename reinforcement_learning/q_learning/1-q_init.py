#!/usr/bin/env python3

import numpy as np

def q_init(env):
    n_states = env.observation_space.n   # nombre d'états
    n_actions = env.action_space.n       # nombre d'actions
    return np.zeros((n_states, n_actions))
