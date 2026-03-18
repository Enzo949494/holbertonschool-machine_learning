#!/usr/bin/env python3
"""TD(lambda) algorithm for value estimation"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm for value estimation."""
    for _ in range(episodes):
        state, _ = env.reset()
        E = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Pénalise les trous
            if terminated and reward == 0:
                reward = -1

            delta = reward + gamma * V[next_state] * (1 - int(terminated)) - V[state]

            E[state] += 1
            V += alpha * delta * E
            E *= gamma * lambtha

            state = next_state
            if terminated or truncated:
                break

    return V
