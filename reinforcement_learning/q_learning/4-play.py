#!/usr/bin/env python3
"""Module for playing a trained Q-learning agent on FrozenLake."""
import numpy as np


def play(env, Q, max_steps=100):
    """Play an episode using the trained Q-table.

    Args:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: total rewards for the episode
        rendered_outputs: list of rendered board states at each step
    """
    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = []

    rendered_outputs.append(env.render())

    for step in range(max_steps):
        action = np.argmax(Q[state])  # toujours exploiter, jamais explorer

        new_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rendered_outputs.append(env.render())

        total_rewards += reward
        state = new_state

        if done:
            break

    return total_rewards, rendered_outputs
