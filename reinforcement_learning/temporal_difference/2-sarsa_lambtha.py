#!/usr/bin/env python3
"""SARSA(lambda) algorithm"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select the next action using the epsilon_greedy strategy."""
    p = np.random.uniform(0, 1)
    if p > epsilon:
        return np.argmax(Q[state])
    return np.random.randint(Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """Performs the SARSA(lambda) algorithm."""
    for episode in range(episodes):
        state, _ = env.reset()
        e = np.zeros(Q.shape)
        action = epsilon_greedy(Q, state, epsilon=epsilon)

        for step in range(max_steps):
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon=epsilon)

            delta = reward + gamma * Q[new_state, new_action] - Q[state, action]

            # Decay toute la trace puis incrément
            e *= gamma * lambtha
            e[state, action] += 1

            Q = Q + alpha * delta * e

            if terminated or truncated:
                break
            state = new_state
            action = new_action

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
