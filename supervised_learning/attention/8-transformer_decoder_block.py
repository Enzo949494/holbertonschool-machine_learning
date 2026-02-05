#!/usr/bin/env python3
"""
Transformer Decoder Block Implementation

This module implements a transformer decoder block that combines
masked self-attention, encoder-decoder cross-attention, and a
feed-forward network, with layer normalization
and dropout for regularization.
"""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Transformer decoder block with masked self-attention,
    encoder-decoder attention, and FFN.

    Includes layer normalization and dropout for regularization.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initialize the DecoderBlock.

        Args:
            dm (int): The dimensionality of the model
            h (int): The number of heads for multi-head attention
            hidden (int): The dimensionality of the hidden layer in the
                          feed-forward network
            drop_rate (float): Dropout rate (default: 0.1)
        """
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)  # masked self-attention
        self.mha2 = MultiHeadAttention(dm, h)  # encoder-decoder attention
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass of the decoder block.

        Applies masked self-attention, encoder-decoder cross-attention,
        and a feed-forward network, with residual connections and
        layer normalization at each step.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, seq_len, dm)
            encoder_output (tf.Tensor): Output from the encoder of shape
                                        (batch, seq_len, dm)
            training (bool): Whether the model is in training mode
                             (affects dropout)
            look_ahead_mask (tf.Tensor): Mask for preventing attention
                                         to future tokens
            padding_mask (tf.Tensor): Mask for masking padding positions

        Returns:
            tf.Tensor: Output tensor of shape (batch, seq_len, dm)
        """
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)  # residual

        # 2. Encoder-Decoder attention (mha2)
        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)  # residual

        # 3. FFN
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)  # residual

        return out3
