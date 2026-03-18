#!/usr/bin/env python3
"""Module algorithme SARSA(lambda) pour l'apprentissage par renforcement.

SARSA(lambda), un algorithme Temporal Difference avec traces d'éligibilité
pour l'apprentissage par renforcement. Contrairement aux algorithmes
d'évaluation de politique, SARSA apprend une fonction de valeur action-état(Q)
et améliore la politique de manière on-policy avec une stratégie epsilon-greedy
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1,
                  min_epsilon=0.1, epsilon_decay=0.05):
    """Effectue l'algo SARSA(lambda) pour l'apprentissage par renforcement.

    Implémente l'apprentissage SARSA (State-Action-Reward-State-Action)
    avec traces d'éligibilité pour optimiser une fonction de valeur
    action-état (Q). L'algorithme utilise une stratégie epsilon-greedy
    et met à jour les Q-valeurs en temps réel basé sur l'action
    réellement prise (on-policy).

    Args:
        env: L'environnement Gymnasium avec les méthodes reset() et step().
        Q (np.ndarray): Matrice initiale des valeurs action-état (Q-values)
                        à mettre à jour. Shape: (nb_états, nb_actions).
        lambtha (float): Paramètre de trace d'éligibilité (décay géométrique).
                        Entre 0 (TD classique) et 1 (Monte Carlo).
        episodes (int, optional): Nombre d'épisodes à simuler. Par défaut 5000.
        max_steps (int, optional): Nombre maximal d'étapes par épisode.
                                  Par défaut 100.
        alpha (float, optional): Taux d'apprentissage. Par défaut 0.1.
        gamma (float, optional): Facteur d'actualisation. Par défaut 0.99.
        epsilon (float, optional): Paramètre de probabilité d'exploration
                                   initial pour epsilon-greedy. Par défaut 1.0.
        min_epsilon (float, optional): Valeur minimale d'epsilon après
                                       décroissance Par défaut 0.1.
        epsilon_decay (float, optional): Taux de décay d'epsilon par épisode.
                                        Par défaut 0.05.

    Returns:
        np.ndarray: matrice des valeurs action-état (Q-values) mise à jour."""
    def epsilon_greedy(state):
        if np.random.uniform() > epsilon:
            return np.argmax(Q[state])
        return env.action_space.sample()

    for episode in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(state)
        e = np.zeros(Q.shape)

        for step in range(max_steps):
            new_state, reward, terminated, truncated, _ = env.step(action)
            new_action = epsilon_greedy(new_state)

            delta = reward + gamma * Q[
                new_state, new_action] - Q[state, action]
            e = gamma * lambtha * e
            e[state, action] += 1
            Q = Q + alpha * delta * e

            if terminated or truncated:
                break
            state = new_state
            action = new_action

        # Décroissance epsilon après chaque épisode
        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
