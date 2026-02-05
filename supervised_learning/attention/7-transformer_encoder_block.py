#!/usr/bin/env python3
"""
Transformer Encoder Block Implementation

This module implements a transformer encoder block that combines
multi-head attention with a feed-forward network, including layer
normalization and dropout for regularization.
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Transformer encoder block combining multi-head
    self-attention and feed-forward network.

    Includes layer normalization and dropout for regularization.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the EncoderBlock.

        Args:
            dm (int): The dimensionality of the model
            h (int): The number of heads for multi-head attention
            hidden (int): The dimensionality of the hidden layer in
                          the feed-forward network
            drop_rate (float): Dropout rate (default: 0.1)
        """
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass of the encoder block.

        Applies multi-head self-attention followed by a feed-forward network,
        with residual connections and layer normalization at each step.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, dm)
            training (bool): Whether the model is in training mode
                             (affects dropout)
            mask (tf.Tensor, optional): Attention mask for masking
                                        certain positions

        Returns:
            tf.Tensor: Output tensor of shape (batch, seq_len, dm)
        """
        attn_output, _ = self.mha(x, x, x, mask)  # self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)   # residual

        # 2. FFN + residual + norm
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # residual

        return out2
