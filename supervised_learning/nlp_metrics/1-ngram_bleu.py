#!/usr/bin/env python3
"""
Module for calculating n-gram BLEU score
"""

from collections import Counter
import math


def ngram_bleu(references, sentence, n):
    """
    Calculate n-gram BLEU score for a sentence

    Args:
        references: list of reference translations, each is a list of words
        sentence: list of words in the model's proposed sentence
        n: size of the n-gram to use for evaluation

    Returns:
        float: the n-gram BLEU score
    """
    # Return 0 if sentence is too short for n-grams
    sentence_len = len(sentence)
    if sentence_len < n:
        return 0.0

    # Extract n-grams from the candidate sentence
    sentence_ngrams = [tuple(sentence[i:i + n])
                       for i in range(sentence_len - n + 1)]
    sentence_counts = Counter(sentence_ngrams)

    # Find maximum occurrences of each n-gram across all references
    max_ref_counts = {}
    for ngram in sentence_counts:
        max_count = 0
        for ref in references:
            if len(ref) >= n:
                ref_ngrams = [tuple(ref[i:i + n])
                              for i in range(len(ref) - n + 1)]
                max_count = max(max_count, ref_ngrams.count(ngram))
            max_ref_counts[ngram] = max_count

    # Calculate clipped precision
    clipped_count = sum(min(sentence_counts[ngram],
                            max_ref_counts[ngram])
                        for ngram in sentence_counts)
    precision = clipped_count / len(sentence_ngrams)

    # Find closest reference length and apply brevity penalty
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - sentence_len))
    bp = (1.0 if sentence_len >= closest_ref_len
          else math.exp(1 - closest_ref_len / sentence_len))

    # Return BLEU score = penalty × precision
    return bp * precision
