#!/usr/bin/env python3
"""Module for converting gensim Word2Vec models to Keras Embedding layers."""

import tensorflow as tf


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a keras Embedding layer.
    
    Args:
        model: trained gensim word2vec model
    
    Returns:
        trainable keras Embedding layer
    """
    # Get the embedding vectors and vocabulary from the gensim model
    weights = model.wv.vectors
    vocab_size = len(model.wv)
    embedding_dim = model.wv.vector_size
    
    # Create the Keras Embedding layer with the weights
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights],
        trainable=True
    )
    
    return embedding_layer
