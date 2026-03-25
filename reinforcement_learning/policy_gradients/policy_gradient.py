#!/usr/bin/env python3
"""
Policy Gradient Module

This module contains functions for computing policy gradients
in reinforcement learning algorithms.
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy (action probabilities) using softmax.

    Parameters
    ----------
    matrix : np.ndarray
        State vector with shape (1, state_size)
    weight : np.ndarray
        Weight matrix with shape (state_size, action_size)

    Returns
    -------
    np.ndarray
        Action probability distribution with shape (1, action_size)
    """
    # Calcule les scores logits en multipliant l'état par les poids
    z = matrix @ weight

    # Soustrait le max pour la stabilité numérique (évite overflow)
    z -= np.max(z, axis=1, keepdims=True)

    # Applique l'exponentielle pour obtenir les valeurs brutes
    exp_z = np.exp(z)

    # Normalise pour obtenir une distribution de probabilité (somme = 1)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Calculates the policy gradient for a given state and weight matrix.

    Parameters
    ----------
    state : np.ndarray
        The current state as a 1D array, shape (state_size,)
    weight : np.ndarray
        The weight matrix with shape (state_size, action_size)

    Returns
    -------
    tuple
        - action : int
            The action sampled from the policy
        - grad : np.ndarray
            The policy gradient with shape (state_size, action_size)
    """
    # Reshape l'état pour qu'il soit compatible avec la fonction policy
    probs = policy(state.reshape(1, -1), weight)

    # Sélectionne une action en fonction des probabilités du réseau
    action = np.random.choice(len(probs[0]), p=probs[0])

    # Crée un vecteur one-hot pour l'action sélectionnée
    one_hot = np.zeros_like(probs[0])
    one_hot[action] = 1

    # Calcule le gradient:
    # outer product de l'état et de (récompense - probabilité)
    grad = state[:, np.newaxis] * (one_hot - probs[0])

    return action, grad
