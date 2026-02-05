#!/usr/bin/env python3

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Reshape: (batch, seq, dm) → (batch, h, seq, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, h, seq, depth)

    def call(self, Q, K, V, mask):
        batch_size = tf.shape(Q)[0]
        
        # Projections
        Q = self.Wq(Q)  # (batch, seq_q, dm)
        K = self.Wk(K)  # (batch, seq_v, dm)
        V = self.Wv(V)  # (batch, seq_v, dm)
        
        # Split heads
        Q = self.split_heads(Q, batch_size)  # (batch, h, seq_q, depth)
        K = self.split_heads(K, batch_size)  # (batch, h, seq_v, depth)
        V = self.split_heads(V, batch_size)  # (batch, h, seq_v, depth)
        
        # SDP par tête
        output, weights = sdp_attention(Q, K, V, mask)
        
        # Concat heads: (batch, h, seq_q, depth) → (batch, seq_q, dm)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.dm))
        
        # Linear final
        output = self.linear(output)
        
        return output, weights
