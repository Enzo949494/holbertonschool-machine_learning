#!/usr/bin/env python3
"""
Module for epsilon-greedy action selection strategy.

This module implements the epsilon-greedy algorithm, a fundamental
exploration-exploitation trade-off strategy used in reinforcement
learning to balance between exploring new actions
and exploiting known good actions.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Select an action using the epsilon-greedy strategy.

    The epsilon-greedy strategy balances exploration and exploitation:
    - With probability epsilon: choose a random action (exploration)
    - With probability 1-epsilon: choose the action with highest Q-value
                                  (exploitation)

    Args:
        Q (numpy.ndarray): The Q-table shape (n_states, n_actions) containing
            Q-values for each state-action pair.
        state (int): The current state index.
        epsilon (float): The exploration rate (0 <= epsilon <= 1).
            - epsilon = 0: pure exploitation (always take best action)
            - epsilon = 1: pure exploration (always take random action)
            - 0 < epsilon < 1: balance between exploration and exploitation

    Returns:
        int: The selected action index (between 0 and n_actions-1).
    """
    p = np.random.uniform(0, 1)  # tire un nombre entre 0 et 1

    if p < epsilon:
        # EXPLORATION : action aléatoire
        return np.random.randint(0, Q.shape[1])
    else:
        # EXPLOITATION : meilleure action connue
        return np.argmax(Q[state])
