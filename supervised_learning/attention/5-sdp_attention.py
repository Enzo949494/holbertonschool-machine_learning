#!/usr/bin/env python3

import tensorflow as tf

def sdp_attention(Q, K, V, mask=None):
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
