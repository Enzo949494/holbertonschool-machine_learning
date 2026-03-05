#!/usr/bin/env python3
"""
Module for initializing Q-learning Q-table.

This module provides utilities for initializing the Q-table used in
Q-learning reinforcement learning algorithms.
"""

import numpy as np


def q_init(env):
    """
    Initialize a Q-table with zeros.

    The Q-table is a matrix where rows represent states and columns
    represent actions. It stores the estimated Q-values (expected
    cumulative rewards) for each state-action pair.

    Args:
        env (gymnasium.Env): A gymnasium environment containing:
            - observation_space.n: The number of states in the environment
            - action_space.n: The number of actions available

    Returns:
        numpy.ndarray: A Q-table of shape (n_states, n_actions) initialized
            with zeros. Each element represents the Q-value for a
            state-action pair.
    """
    n_states = env.observation_space.n   # nombre d'états
    n_actions = env.action_space.n       # nombre d'actions
    return np.zeros((n_states, n_actions))
