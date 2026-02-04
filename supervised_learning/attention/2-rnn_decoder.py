#!/usr/bin/env python3
"""RNN Decoder for machine translation"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class"""
    
    def __init__(self, vocab, embedding, units, batch):
        """
        Constructor for RNN Decoder
        
        Args:
            vocab: size of the output vocabulary
            embedding: dimensionality of the embedding vector
            units: number of hidden units in the RNN cell
            batch: batch size
        """
        super(RNNDecoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Forward pass of the decoder
        
        Args:
            x: tensor of shape (batch, 1) - previous word index
            s_prev: tensor of shape (batch, units) - previous decoder hidden state
            hidden_states: tensor of shape (batch, input_seq_len, units) - encoder outputs
            
        Returns:
            y: tensor of shape (batch, vocab) - output word as one-hot vector
            s: tensor of shape (batch, units) - new decoder hidden state
        """
        attention = SelfAttention(self.units)
        
        # Calculate attention context vector
        context, _ = attention(s_prev, hidden_states)
        
        # Embed input word
        x = self.embedding(x)
        
        # Concatenate context vector with embedded input (context first)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        
        # Pass through GRU (without initial_state!)
        x, s = self.gru(x)
        
        # Reshape to remove sequence dimension
        x = tf.reshape(x, (-1, x.shape[2]))
        
        # Pass through dense layer
        y = self.F(x)
        
        return y, s