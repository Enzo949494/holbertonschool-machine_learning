#!/usr/bin/env python3
"""
Module for loading and managing FrozenLake environment from gymnasium.

This module provides utilities for creating and configuring the FrozenLake-v1
environment for reinforcement learning tasks.
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Load and create a FrozenLake-v1 gymnasium environment.

    Args:
        desc (list of list of str, optional): A custom environment map.
            Default is None, which uses the default map.
        map_name (str, optional): A standard map name ('4x4' or '8x8').
            Default is None, which uses '8x8'.
        is_slippery (bool, optional): Whether the environment is slippery.
            If True, the agent may not move in the intended direction.
            Default is False.

    Returns:
        gymnasium.Env: The FrozenLake-v1 environment instance.
    """
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env
