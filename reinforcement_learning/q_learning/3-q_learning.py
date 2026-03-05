#!/usr/bin/env python3
"""Module for Q-learning training on FrozenLake environment."""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs Q-learning on a FrozenLake environment.

    Args:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: the updated Q-table
        total_rewards: list of rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated and reward == 0:
                reward = -1

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            state = new_state
            episode_reward += reward

            if done:
                break

        epsilon = max(min_epsilon, epsilon - epsilon_decay)
        total_rewards.append(episode_reward)

    return Q, total_rewards
