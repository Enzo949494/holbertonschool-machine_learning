#!/usr/bin/env python3
"""Effectue l'algorithme SARSA λ pour estimer la fonction valeur."""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Select the next action using the epsilon_greedy strategy."""
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state, :])
    return action


def sarsa_lambtha(
    env,
    Q,
    lambtha,
    episodes=5000,
    max_steps=100,
    alpha=0.1,
    gamma=0.99,
    epsilon=1,
    min_epsilon=0.1,
    epsilon_decay=0.05,
):
    """Effectue l'algorithme SARSA λ pour estimer la fonction valeur."""
    initial_epsilon = epsilon
    n_states, n_actions = Q.shape
    E = np.zeros((n_states, n_actions))

    for episode in range(episodes):
        E.fill(0)
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        for step in range(max_steps):
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)

            delta = (
                reward + gamma * Q[new_state, new_action] - Q[state, action]
            )

            E[state, action] += 1
            Q += alpha * delta * E
            E *= gamma * lambtha

            state, action = new_state, new_action

            if terminated or truncated:
                break

        epsilon = max(
            min_epsilon,
            min_epsilon + (initial_epsilon - min_epsilon)
            * np.exp(-epsilon_decay * episode),
        )

    return Q