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

    def call(self, x, s_prev, hidden_states):
        # 1. Embed
        x_emb = self.embedding(x)
        
        # 2. Attention
        attention = SelfAttention(self.gru.units)
        context, _ = attention(s_prev, hidden_states)
        
        # 3. Concat (reshape x_emb si besoin)
        concat_input = tf.concat([context, tf.squeeze(x_emb, 1)], axis=-1)
        
        # 4. GRU
        _, s = self.gru(tf.expand_dims(concat_input, 1), initial_state=s_prev)

        
        # 5. Logits
        y = self.F(s)
        
        return y, s
