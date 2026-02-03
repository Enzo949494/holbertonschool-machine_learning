#!/usr/bin/env python3

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=False,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        # 1. Embedding of input word
        x_emb = self.embedding(x)  # (32, 1, 128)
        
        # 2. Attention to get context vector
        context, _ = self.attention(s_prev, hidden_states)  # (32, 256)
        
        # 3. Concatenate context with x_emb (context first)
        context_exp = tf.expand_dims(context, 1)  # (32, 1, 256)
        concat_input = tf.concat([context_exp, x_emb], axis=-1)  # (32, 1, 384)
        
        # 4. Pass through GRU with initial state
        output, s = self.gru(concat_input, initial_state=s_prev)  # output:(32,256), s:(32,256)
        
        # 5. Pass GRU output through Dense layer
        y = self.F(output)  # (32, vocab)
        
        return y, s
