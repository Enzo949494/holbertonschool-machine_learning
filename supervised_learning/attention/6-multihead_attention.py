#!/usr/bin/env python3
"""
Multi-Head Attention Layer Implementation

This module implements a multi-head attention mechanism using TensorFlow,
which allows the model to attend to information from different representation
subspaces at different positions.
"""

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-head attention layer for scaled
       dot-product attention with multiple heads."""
    def __init__(self, dm, h):
        """
        Initialize the MultiHeadAttention layer.

        Args:
            dm (int): The dimensionality of the model
            h (int): The number of heads for multi-head attention
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Reshape and transpose the input to separate multiple attention heads.

        Transforms from shape (batch, seq, dm) to (batch, h, seq, depth),
        where depth = dm // h.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, dm)
            batch_size (int): The batch size

        Returns:
            tf.Tensor: Reshaped and transposed tensor of shape
                       (batch, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, h, seq, depth)

    def call(self, Q, K, V, mask):
        """
        Apply multi-head attention to the input.

        Args:
            Q (tf.Tensor): Query tensor of shape (batch, seq_q, dm)
            K (tf.Tensor): Key tensor of shape (batch, seq_v, dm)
            V (tf.Tensor): Value tensor of shape (batch, seq_v, dm)
            mask (tf.Tensor): Attention mask (optional)

        Returns:
            tuple: A tuple containing:
                - output (tf.Tensor): Attention output shape (batch, seq_q, dm)
                - weights (tf.Tensor): Attention weights
        """
        batch_size = tf.shape(Q)[0]

        # Projections
        Q = self.Wq(Q)  # (batch, seq_q, dm)
        K = self.Wk(K)  # (batch, seq_v, dm)
        V = self.Wv(V)  # (batch, seq_v, dm)

        # Split heads
        Q = self.split_heads(Q, batch_size)  # (batch, h, seq_q, depth)
        K = self.split_heads(K, batch_size)  # (batch, h, seq_v, depth)
        V = self.split_heads(V, batch_size)  # (batch, h, seq_v, depth)

        # SDP par tête
        output, weights = sdp_attention(Q, K, V, mask)

        # Concat heads: (batch, h, seq_q, depth) → (batch, seq_q, dm)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.dm))

        # Linear final
        output = self.linear(output)

        return output, weights
