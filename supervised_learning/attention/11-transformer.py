#!/usr/bin/env python3
"""
Transformer Network Implementation

This module implements a complete transformer model combining
an encoder and decoder for sequence-to-sequence tasks with
self-attention mechanisms.
"""

import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Complete transformer model with encoder and decoder components
    for sequence-to-sequence tasks.
    """
    def __init__(
            self, N, dm, h, hidden, input_vocab, target_vocab,
            max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Initialize the Transformer.

        Args:
            N (int): The number of encoder and decoder blocks
            dm (int): The dimensionality of the model
            h (int): The number of heads for multi-head attention
            hidden (int): The dimensionality of the hidden layer in
                          the feed-forward networks
            input_vocab (int): The size of the input vocabulary
            target_vocab (int): The size of the target vocabulary
            max_seq_input (int): The maximum sequence length for inputs
            max_seq_target (int): The maximum sequence length for targets
            drop_rate (float): Dropout rate (default: 0.1)
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
            self, inputs, target, training, encoder_mask, look_ahead_mask,
            decoder_mask):
        """
        Forward pass of the transformer.

        Encodes the input sequence and decode to produce target sequence logit.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch, input_seq_len)
                               containing input token indices
            target (tf.Tensor): Target tensor of shape (batch, target_seq_len)
                               containing target token indices
            training (bool): Whether the model is in training mode
                            (affects dropout)
            encoder_mask (tf.Tensor, optional): Padding mask for the encoder
            look_ahead_mask (tf.Tensor, optional): Look-ahead mask for
                                                   the decoder self-attention
            decoder_mask (tf.Tensor, optional): Padding mask for
                                               the decoder cross-attention

        Returns:
            tf.Tensor: Transformer output of shape (batch, target_seq_len,
                      target_vocab) containing logits for each target token
        """
        # Encode the inputs
        encoder_output = self.encoder(inputs, training, encoder_mask)

        # Decode with encoder-decoder attention
        decoder_output = self.decoder(
            target, encoder_output, training, look_ahead_mask, decoder_mask)

        # Project to vocabulary
        output = self.linear(decoder_output)

        return output
