#!/usr/bin/env python3
"""
Module pour l'encodeur RNN utilisé dans les modèles d'attention.

Ce module implémente une couche encodeur RNN basée sur GRU qui transforme
une séquence d'entrée en représentations d'état caché.
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Encodeur RNN basé sur GRU pour les modèles de séquence à séquence.

    Cette classe encapsule un encodeur qui prend des séquences en entrée,
    les projette dans un espace d'embedding, et les traite avec une couche GRU.
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Initialise l'encodeur RNN avec les couches embedding et GRU.

        Args:
            vocab (int): Taille du vocabulaire pour la couche embedding
            embedding (int): Dimension de l'espace d'embedding
            units (int): Nombre d'unités dans la couche GRU
            batch (int): Taille du batch pour l'état initial
        """
        super(RNNEncoder, self).__init__()

        # Stockage des paramètres
        self.batch = batch
        self.units = units

        # Couche d'embedding pour transformer les indices en vecteurs
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding)

        # Couche GRU pour traiter la séquence d'embedding
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,  # Retourner la sortie complète
            return_state=True,      # Retourner l'état caché final
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Crée un état caché initial de zéros.

        L'état caché initial est utilisé au premier pas de temps du décodeur.

        Returns:
            tf.Tensor: Tenseur de forme (batch, units) rempli de zéros
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Traite une séquence d'entrée et retourne les sorties et l'état final.

        Args:
            x (tf.Tensor): Séquence d'entrée de forme (batch, seq_length)
            initial (tf.Tensor): État caché initial de forme (batch, units)

        Returns:
            tuple: (outputs, hidden)
                - outputs: Sorties de tous les pas de temps
                           (batch, seq_length, units)
                - hidden: État caché final (batch, units)
        """
        # Convertir les indices en vecteurs d'embedding
        x = self.embedding(x)

        # Traiter la séquence d'embedding avec la GRU
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
