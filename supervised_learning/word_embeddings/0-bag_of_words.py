#!/usr/bin/env python3
"""Module for creating bag of words embeddings."""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix.

    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words (if None, use all words from sentences)

    Returns:
        embeddings: numpy array of shape (s, f) containing the embeddings
        features: numpy array of the features (vocabulary words) used
    """
    if vocab is None:
        # Étape 1 : extraire vocabulaire
        all_words = []
        for sentence in sentences:
            words = [w for w in re.findall
                     (r'[a-z]+', sentence.lower()) if len(w) > 1]
            all_words.extend(words)
        vocab = sorted(set(all_words))

    # Étape 2 : créer matrice
    n_sentences = len(sentences)
    n_features = len(vocab)
    embeddings = np.zeros((n_sentences, n_features), dtype=int)

    # Étape 3 : remplir matrice
    for i, sentence in enumerate(sentences):
        words = [w for w in re.findall
                 (r'[a-z]+', sentence.lower()) if len(w) > 1]
        for word in words:
            if word in vocab:
                col_idx = vocab.index(word)
                embeddings[i, col_idx] += 1

    return embeddings, np.array(vocab)
