#!/usr/bin/env python3
"""
Transformer Decoder Implementation

This module implements a transformer decoder that stacks multiple decoder blocks
with embeddings and positional encodings to process target sequences while
attending to encoder outputs.
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Transformer decoder that combines embeddings, positional encodings,
    and multiple decoder blocks for sequence decoding with cross-attention
    to encoder outputs.
    """
    def __init__(
            self, N, dm, h, hidden, target_vocab, max_seq_len, drop_rate=0.1):
        """
        Initialize the Decoder.

        Args:
            N (int): The number of decoder blocks
            dm (int): The dimensionality of the model
            h (int): The number of heads for multi-head attention
            hidden (int): The dimensionality of the hidden layer in
                          the feed-forward network
            target_vocab (int): The size of the target vocabulary
            max_seq_len (int): The maximum sequence length possible
            drop_rate (float): Dropout rate (default: 0.1)
        """
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(
            dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(
            self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass of the decoder.

        Applies embeddings, positional encodings, dropout,
        and all decoder blocks with encoder cross-attention.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, target_seq_len)
                          containing target token indices
            encoder_output (tf.Tensor): Output from the encoder of shape
                                       (batch, input_seq_len, dm)
            training (bool): Whether the model is in training mode
                            (affects dropout)
            look_ahead_mask (tf.Tensor, optional): Mask for preventing
                                                   attention to future tokens
            padding_mask (tf.Tensor, optional): Mask for masking padding
                                               positions

        Returns:
            tf.Tensor: Decoder output of shape (batch, target_seq_len, dm)
        """
        # Embedding
        x = self.embedding(x)  # (batch, target_seq_len, dm)

        # Positional encoding
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len, :], tf.float32)

        # Dropout
        x = self.dropout(x, training=training)

        # N blocks with encoder-decoder attention
        for block in self.blocks:
            x = block(
                x, encoder_output, training, look_ahead_mask, padding_mask)

        return x
