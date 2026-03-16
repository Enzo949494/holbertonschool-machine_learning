#!/usr/bin/env python3

import numpy as np

def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Pénalise les trous avec -1
            if terminated and reward == 0:
                reward = -1

            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        G = 0
        for s, reward in reversed(episode):
            G = reward + gamma * G
            V[s] = V[s] + alpha * (G - V[s])

    return V
