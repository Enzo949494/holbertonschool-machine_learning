#!/usr/bin/env python3
"""Module implémentant l'algorithme TD(lambda) pour
   l'estimation de fonction de valeur.

Ce module contient l'implémentation de l'algorithme Temporal Difference avec
traces d'éligibilité (TD(lambda)) pour l'apprentissage par renforcement. Cet
algorithme combine les avantages du Monte Carlo et du TD classique en utilisant
des traces d'éligibilité contrôlées par le paramètre lambda.
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Effectue l'algorithme TD(lambda) pour l'estimation
       d'une fonction de valeur.

    Implémente apprentissage Temporal Difference avec traces d'éligibilité pour
    évaluer une fonction de valeur états (V) selon une politique donnée.
    L'algorithme utilise les traces d'éligibilité pour propager
    les mises à jour de valeur sur plusieurs états au cours d'un épisode.

    Args:
        env: L'environnement Gymnasium avec les méthodes reset() et step().
        V (np.ndarray): Vecteur initial des valeurs d'états à mettre à jour.
        policy (callable): Fonction de politique prenant un état et retournant
                          une action.
        lambtha (float): Paramètre de trace d'éligibilité (décay géométrique).
                        Entre 0 (TD classique) et 1 (Monte Carlo).
        episodes (int, optional): Nombre d'épisodes à simuler. Par défaut 5000.
        max_steps (int, optional): Nombre maximal d'étapes par épisode.
                                  Par défaut 100.
        alpha (float, optional): Taux d'apprentissage. Par défaut 0.1.
        gamma (float, optional): Facteur d'actualisation. Par défaut 0.99.

    Returns:
        np.ndarray: Le vecteur des valeurs d'états mis à jour."""
    for episode in range(episodes):
        state, _ = env.reset()
        e = np.zeros(V.shape)

        for step in range(max_steps):
            action = policy(state)
            new_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[new_state] - V[state]
            e = gamma * lambtha * e
            e[state] += 1
            V = V + alpha * delta * e

            if terminated or truncated:
                break
            state = new_state

    return V
