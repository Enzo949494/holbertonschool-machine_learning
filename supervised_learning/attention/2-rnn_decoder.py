#!/usr/bin/env python3
"""RNN Decoder for machine translation"""

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class"""
    
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        # Calculate attention
        context, _ = self.attention(s_prev, hidden_states)
        
        # Embed input
        x = self.embedding(x)
        
        # Concatenate context with embedding
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        
        # GRU
        output, s = self.gru(x, initial_state=s_prev)
        output = output[:, -1, :]
        
        # Dense
        y = self.F(output)
        
        return y, s
    