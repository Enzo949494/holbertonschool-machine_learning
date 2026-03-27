#!/usr/bin/env python3
"""
Training Module for Policy Gradient

This module implements the training loop for a policy gradient reinforcement
learning agent using Monte-Carlo returns to optimize the policy network.
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Train a policy gradient agent using Monte-Carlo returns.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to train the agent on
    nb_episodes : int
        Number of episodes to train for
    alpha : float, optional
        Learning rate for weight updates (default: 0.000045)
    gamma : float, optional
        Discount factor for computing returns (default: 0.98)

    Returns
    -------
    list
        List of episode scores (sum of rewards per episode)

    Notes
    -----
    The training follows the REINFORCE algorithm:
    1. Play a complete episode and collect gradients and rewards
    2. Compute discounted cumulative returns (Monte-Carlo)
    3. Update weights using gradient ascent
    """
    # Initialisation des poids aléatoires (state_size=4, action_size=2)
    weight = np.random.rand(4, 2)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        episode_gradients = []
        episode_rewards = []
        terminated = False

        # -- Phase 1 : jouer un épisode complet --
        while not terminated:
            action, grad = policy_gradient(state, weight)
            new_state, reward, terminated, truncated, _ = env.step(action)

            episode_gradients.append(grad)
            episode_rewards.append(reward)
            state = new_state
            terminated = terminated or truncated

        # -- Phase 2 : calculer les retours cumulés (Monte-Carlo) --
        G = 0
        returns = []
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)

        # -- Phase 3 : mettre à jour les poids --
        for grad, G_t in zip(episode_gradients, returns):
            weight += alpha * grad * G_t

        score = sum(episode_rewards)
        scores.append(score)
        print(f"Episode: {episode} Score: {score}")

    return scores
