#!/usr/bin/env python3
"""
Scaled Dot-Product Attention Module

This module implements the scaled dot-product attention mechanism, which is a
fundamental component of the transformer architecture. It computes attention
weights based on query, key, and value matrices and applies an optional mask.
"""

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculate scaled dot-product attention.

    Implements the attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(dk))V

    Args:
        Q (tf.Tensor): Query matrix of shape (batch_size, seq_len_q, dk)
        K (tf.Tensor): Key matrix of shape (batch_size, seq_len_k, dk)
        V (tf.Tensor): Value matrix of shape (batch_size, seq_len_v, dv)
        mask (tf.Tensor, optional): Mask tensor for blocking attention
                                    to certain positions.
                                    Defaults to None

    Returns:
        tuple: A tuple containing:
            - output (tf.Tensor): Attention output of shape
                                  (batch_size, seq_len_q, dv)
            - attention_weights (tf.Tensor): Attention weights of shape
                                             (batch_size, seq_len_q, seq_len_k)
    """
    # 1. Scores: Q @ K^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # 2. Scale: / sqrt(dk)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # 3. Mask si fourni
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 4. Softmax → weights
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # 5. Weights @ V → output
    output = tf.matmul(attention_weights, V)

    return output, attention_weights
