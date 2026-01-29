#!/usr/bin/env python3
"""
Module for calculating unigram BLEU score
"""

from collections import Counter
import math


def uni_bleu(references, sentence):
    """
    Calculate unigram BLEU score for a sentence

    Args:
        references: list of reference translations, each is a list of words
        sentence: list of words in the model's proposed sentence

    Returns:
        float: the unigram BLEU score
    """
    # Return 0 if sentence is empty
    sentence_len = len(sentence)
    if sentence_len == 0:
        return 0.0

    # Count occurrences of each word in the candidate sentence
    sentence_counts = Counter(sentence)

    # Find the maximum occurrences of each word across all references
    max_ref_counts = {}
    for word in sentence_counts:
        max_count = 0
        for ref in references:
            max_count = max(max_count, ref.count(word))
        max_ref_counts[word] = max_count

    # Calcul clipped precision (min between candidate count & max ref count)
    clipped_count = sum(min(sentence_counts[word], max_ref_counts[word])
                        for word in sentence_counts)
    precision = clipped_count / sentence_len

    # Find closest reference length and apply brevity penalty
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - sentence_len))
    bp = (1.0 if sentence_len >= closest_ref_len
          else math.exp(1 - closest_ref_len / sentence_len))

    # Return BLEU score = penalty × precision
    return bp * precision
