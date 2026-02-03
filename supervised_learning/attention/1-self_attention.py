#!/usr/bin/env python3
"""
Module pour le mécanisme d'attention (Self-Attention) en séquence à séquence.

Ce module implémente une couche d'attention qui calcule l'importance relative
de chaque élément d'une séquence d'entrée par rapport à un état caché donné.
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Mécanisme d'attention pour les modèles de séquence à séquence.

    Cette couche calcule un vecteur de contexte pondéré basé sur:
    - L'état précédent du décodeur (s_prev)
    - Les états cachés de l'encodeur (hidden_states)

    Les poids d'attention sont calculés via un réseau de
    neurones simple (Bahdanau attention).
    """
    def __init__(self, units):
        """
        Initialise les couches denses pour calculer les poids d'attention.

        Args:
            units (int): Nombre d'unités pour les couches denses internes
        """
        super(SelfAttention, self).__init__()

        # Couche dense pour transformer l'état précédent du décodeur
        self.W = tf.keras.layers.Dense(units)

        # Couche dense pour transformer les états cachés de l'encodeur
        self.U = tf.keras.layers.Dense(units)

        # Couche dense pour calculer le score d'attention final
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Calcule le vecteur de contexte et les poids d'attention.

        Args:
            s_prev (tf.Tensor): État caché précédent du décodeur (batch, units)
            hidden_states (tf.Tensor): États cachés de l'encodeur
                                       (batch, seq_len, units)

        Returns:
            tuple: (context, weights)
                - context: Vecteur de contexte pondéré (batch, units)
                - weights: Poids d'attention (batch, seq_len, 1)
        """
        # Étendre s_prev pour broadcasting: (batch, 1, units)
        # Permet de combiner avec hidden_states via addition
        s_prev = tf.expand_dims(s_prev, 1)

        # Calcul des scores d'énergie d'attention
        # W(s_prev) + U(hidden_states) puis tanh pour non-linéarité
        # Résultat: (batch, seq_len, 1)
        energy = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))

        # Normaliser les scores avec softmax sur la dimension séquence
        # Les poids somment à 1 pour chaque élément du batch
        weights = tf.nn.softmax(energy, axis=1)

        # Calcul du vecteur de contexte: moyenne pondérée des états cachés
        # weights * hidden_states puis somme sur l'axe séquence
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
