#!/usr/bin/env python3

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.F = tf.keras.layers.Dense(vocab)
        # FIX 1: Attention créée UNE SEULE FOIS au init
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        # 1. Embed
        x_emb = self.embedding(x)  # (32,1,128)
        
        # 2. Attention (mêmes poids !)
        context, _ = self.attention(s_prev, hidden_states)  # (32,256)
        
        # 3. Concat
        x_emb_squeezed = tf.squeeze(x_emb, 1)  # (32,128)
        concat_input = tf.concat([context, x_emb_squeezed], axis=-1)  # (32,384)
        
        # 4. Time dimension
        concat_input = tf.expand_dims(concat_input, 1)  # (32,1,384)
        
        # 5. GRU
        _, s = self.gru(concat_input, initial_state=s_prev)
        
        # 6. Logits
        y = self.F(s)
        
        return y, s
