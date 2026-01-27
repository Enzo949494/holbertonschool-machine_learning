#!/usr/bin/env python3

import numpy as np
import re

def bag_of_words(sentences, vocab=None):
    if vocab is None:
        # Étape 1 : extraire vocabulaire
        all_words = []
        for sentence in sentences:
            words = re.findall(r'[a-z]+', sentence.lower())
            all_words.extend(words)
        vocab = sorted(set(all_words))
    
    # Étape 2 : créer matrice
    n_sentences = len(sentences)
    n_features = len(vocab)
    embeddings = np.zeros((n_sentences, n_features), dtype=int)
    
    # Étape 3 : remplir matrice
    for i, sentence in enumerate(sentences):
        words = re.findall(r'[a-z]+', sentence.lower())
        for word in words:
            if word in vocab:
                col_idx = vocab.index(word)
                embeddings[i, col_idx] += 1
    
    return embeddings, vocab
