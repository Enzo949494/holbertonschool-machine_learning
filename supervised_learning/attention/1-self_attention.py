#!/usr/bin/env python3
"""Self attention layer."""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Attention layer."""
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculate attention weights and context vector."""
        # Étendre s_prev pour broadcasting: (batch, 1, units)
        s_prev = tf.expand_dims(s_prev, 1)

        # Scores: (batch, seq_len, 1)
        energy = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))

        # Weights: softmax sur seq_len
        weights = tf.nn.softmax(energy, axis=1)

        # Context: pondéré
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
