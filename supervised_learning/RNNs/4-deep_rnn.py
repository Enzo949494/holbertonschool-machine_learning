#!/usr/bin/env python3
"""Deep RNN forward"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    rnn_cells: list of RNNCell instances, length l
    X: np.ndarray of shape (t, m, i) with the data
    h_0: np.ndarray of shape (l, m, h) with the initial hidden states

    Returns:
        H: np.ndarray of shape (t+1, l, m, h) with all hidden states
        Y: np.ndarray of shape (t, m, o) with all outputs
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    # sortie dimension o de la dernière couche
    o = rnn_cells[-1].Wy.shape[1]

    # initialisation des tenseurs de sortie
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))

    # état initial
    H[0] = h_0

    # boucle temporelle
    for step in range(t):
        x_t = X[step]
        h_prev_layer = H[step]

        h_next_layer = np.zeros((l, m, h))

        # boucle sur les couches
        for layer, cell in enumerate(rnn_cells):
            h_prev = h_prev_layer[layer]
            h_next, y = cell.forward(h_prev, x_t)
            h_next_layer[layer] = h_next

            # l'entrée de la couche suivante est la sortie cachée de la couche courante
            x_t = h_next

        # stocker les états cachés de toutes les couches pour ce pas de temps
        H[step + 1] = h_next_layer
        # la sortie globale Y vient de la dernière couche
        Y[step] = y

    return H, Y
