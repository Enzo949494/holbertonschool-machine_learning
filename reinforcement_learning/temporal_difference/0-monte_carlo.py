#!/usr/bin/env python3
"""Module pour l'estimation d'une fonction de valeur par Monte Carlo.

Ce module implémente l'algorithme de Monte Carlo pour l'apprentissage par
renforcement (RL). Il estime la fonction de valeur d'états
dans un environnement en utilisant une politique donnée.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Estime une fonction de valeur par l'algorithme de Monte Carlo.

    Implémente l'apprentissage par Monte Carlo pour évaluer une fonction de
    valeur états (V) selon une politique donnée. L'algorithme exécute plusieurs
    épisodes, collecte les récompenses et met à jour les valeurs d'états.

    Args:
        env: L'environnement Gymnasium avec les méthodes reset() et step().
        V (np.ndarray): Vecteur initial des valeurs d'états à mettre à jour.
        policy (callable): Fonction de politique prenant un état et retournant
                          une action.
        episodes (int, optional): Nombre d'épisodes à simuler. Par défaut 5000.
        max_steps (int, optional): Nombre maximal d'étapes par épisode.
                                  Par défaut 100.
        alpha (float, optional): Taux d'apprentissage. Par défaut 0.1.
        gamma (float, optional): Facteur d'actualisation. Par défaut 0.99.

    Returns:
        np.ndarray: Le vecteur des valeurs d'états mis à jour."""
    for episode in range(episodes):
        state = 0
        env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))
            if terminated or truncated:
                break
            state = new_state

        episode_data = np.array(episode_data, dtype=int)
        G = 0
        for state, reward in reversed(episode_data):
            G = reward + gamma * G
            if state not in episode_data[:episode, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
