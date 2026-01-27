#!/usr/bin/env python3
"""Module for creating TF-IDF embeddings."""

import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """Creates a TF-IDF embedding matrix.
    
    Args:
        sentences: list of sentences to analyze
        vocab: list of vocabulary words (if None, use all words from sentences)
    
    Returns:
        embeddings: numpy array of shape (s, f) containing the TF-IDF embeddings
        features: numpy array of the features (vocabulary words) used
    """
    # Extract vocabulary if not provided
    if vocab is None:
        all_words = []
        for sentence in sentences:
            words = [w for w in re.findall(r'[a-z]+', sentence.lower()) if len(w) > 1]
            all_words.extend(words)
        vocab = sorted(list(set(all_words)))
    
    n_sentences = len(sentences)
    n_features = len(vocab)
    
    # Create bag of words matrix for counting
    bow = np.zeros((n_sentences, n_features), dtype=int)
    for i, sentence in enumerate(sentences):
        words = [w for w in re.findall(r'[a-z]+', sentence.lower()) if len(w) > 1]
        for word in words:
            if word in vocab:
                col_idx = vocab.index(word)
                bow[i, col_idx] += 1
    
    # Calculate TF (Term Frequency)
    # TF = word count / total words in sentence
    tf = np.zeros((n_sentences, n_features))
    for i in range(n_sentences):
        total_words = np.sum(bow[i])
        if total_words > 0:
            tf[i] = bow[i] / total_words
    
    # Calculate IDF (Inverse Document Frequency)
    # IDF = log((N + 1) / (df + 1)) + 1
    idf = np.zeros(n_features)
    for j in range(n_features):
        # Count how many documents contain word j
        doc_count = np.sum(bow[:, j] > 0)
        idf[j] = np.log((n_sentences + 1) / (doc_count + 1)) + 1
    
    # Calculate TF-IDF
    tfidf = tf * idf
    
    # L2 normalize each row (document)
    embeddings = np.zeros((n_sentences, n_features))
    for i in range(n_sentences):
        norm = np.linalg.norm(tfidf[i])
        if norm > 0:
            embeddings[i] = tfidf[i] / norm
        else:
            embeddings[i] = tfidf[i]
    
    return embeddings, np.array(vocab)
