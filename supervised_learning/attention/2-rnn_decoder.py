#!/usr/bin/env python3

import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
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
        
        # 3. Concatenate x_emb with context
        context_exp = tf.expand_dims(context, 1)  # (32, 1, 256)
        concat_input = tf.concat([x_emb, context_exp], axis=-1)  # (32, 1, 384)
        
        # 4. Pass through GRU
        output, s = self.gru(concat_input, initial_state=s_prev)  # output:(32,1,256), s:(32,256)
        
        # 5. Squeeze output and pass through Dense layer
        output = tf.squeeze(output, axis=1)  # (32, 256)
        y = self.F(output)  # (32, vocab)
        
        return y, s
