#!/usr/bin/env python3
"""
Transformer Encoder Implementation

This module implements a transformer encoder that stacks multiple encoder blocks
with embeddings and positional encodings to process input sequences.
"""

import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock

class Encoder(tf.keras.layers.Layer):
    """
    Transformer encoder that combines embeddings, positional encodings,
    and multiple encoder blocks for sequence encoding.
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len, drop_rate=0.1):
        """
        Initialize the Encoder.
        
        Args:
            N (int): The number of encoder blocks
            dm (int): The dimensionality of the model
            h (int): The number of heads for multi-head attention
            hidden (int): The dimensionality of the hidden layer in the feed-forward network
            input_vocab (int): The size of the input vocabulary
            max_seq_len (int): The maximum sequence length possible
            drop_rate (float): Dropout rate (default: 0.1)
        """
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass of the encoder.
        
        Applies embeddings, positional encodings, dropout, and all encoder blocks.
        
        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len) containing token indices
            training (bool): Whether the model is in training mode (affects dropout)
            mask (tf.Tensor, optional): Attention mask for masking certain positions
            
        Returns:
            tf.Tensor: Encoder output of shape (batch, input_seq_len, dm)
        """
        x = self.embedding(x)  # (batch, seq_len, dm)
        
        # Positional encoding
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += tf.cast(self.positional_encoding[:seq_len, :], tf.float32)
        
        # Dropout
        x = self.dropout(x, training=training)
        
        # N blocks
        for block in self.blocks:
            x = block(x, training, mask)
        
        return x
