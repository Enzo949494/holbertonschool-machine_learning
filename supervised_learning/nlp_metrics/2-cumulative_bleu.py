#!/usr/bin/env python3
"""
Module for calculating cumulative n-gram BLEU score
"""

from collections import Counter
import math


def cumulative_bleu(references, sentence, n):
    """
    Calculate cumulative n-gram BLEU score for a sentence

    Args:
        references: list of reference translations, each is a list of words
        sentence: list of words in the model's proposed sentence
        n: size of the largest n-gram to use for evaluation

    Returns:
        float: the cumulative n-gram BLEU score
    """
    sentence_len = len(sentence)

    # Find closest reference length and calculate brevity penalty
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda x: abs(x - sentence_len))
    bp = (1.0 if sentence_len >= closest_ref_len
          else math.exp(1 - closest_ref_len / sentence_len))

    # Calculate precision for each n-gram size and average them
    bleu_scores = []
    for i in range(1, n + 1):
        # Return 0 if sentence is too short for n-grams
        if sentence_len < i:
            bleu_scores.append(0.0)
            continue

        # Extract n-grams from the candidate sentence
        sentence_ngrams = [tuple(sentence[j:j + i])
                           for j in range(sentence_len - i + 1)]
        sentence_counts = Counter(sentence_ngrams)

        # Find maximum occurrences of each n-gram across all references
        max_ref_counts = {}
        for ngram in sentence_counts:
            max_count = 0
            for ref in references:
                if len(ref) >= i:
                    ref_ngrams = [tuple(ref[j:j + i])
                                  for j in range(len(ref) - i + 1)]
                    max_count = max(max_count, ref_ngrams.count(ngram))
                max_ref_counts[ngram] = max_count

        # Calculate clipped precision for this n-gram size
        clipped_count = sum(min(sentence_counts[ngram],
                                max_ref_counts[ngram])
                            for ngram in sentence_counts)
        precision = clipped_count / len(sentence_ngrams)
        bleu_scores.append(precision)

    # Return cumulative BLEU = brevity penalty × average precision
    avg_precision = sum(bleu_scores) / n
    return bp * avg_precision
