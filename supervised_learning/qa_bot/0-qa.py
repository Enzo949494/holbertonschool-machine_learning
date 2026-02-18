#!/usr/bin/env python3
"""
Module containing the question_answer function
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
import numpy as np

# Charge le tokenizer une seule fois (global)
tokenizer = BertTokenizer.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad"
)

# Charge le modèle TF Hub une seule fois (global)
model = hub.KerasLayer(
    "https://tfhub.dev/see--/bert-uncased-tf2-qa/1",
    trainable=False
)


def question_answer(question, reference):
    """
    Finds a snippet of text in reference that answers the question

    Args:
        question (str): the question to answer
        reference (str): the reference text to search for the answer

    Returns:
        str: answer snippet extracted from reference, None if  answer no found
    """
    # Tokenise la question + contexte
    inputs = tokenizer(
        question,
        reference,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="tf"
    )

    # Prépare les inputs pour le modèle TF Hub
    model_inputs = [
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"]
    ]

    # Inférence : start/end logits
    outputs = model(model_inputs)
    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()

    # Attention mask
    attention_mask = inputs["attention_mask"][0].numpy()
    mask = attention_mask.copy().astype(float)
    mask[0] = 0  # [CLS]

    # Contexte start
    context_start = np.where(inputs["token_type_ids"][0].numpy() == 1)[0]
    if len(context_start) == 0:
        return None
    sep_idx = context_start[0]

    # Masque logits
    masked_start = start_logits.copy()
    masked_end = end_logits.copy()

    for i in range(len(masked_start)):
        if mask[i] == 0 or i < sep_idx:
            masked_start[i] = -float('inf')
            masked_end[i] = -float('inf')

    # Meilleur span (MAX 15 tokens)
    best_score = -float('inf')
    best_start = None
    best_end = None

    for start in range(len(masked_start)):
        if masked_start[start] == -float('inf'):
            continue
        for end in range(start, min(start + 15, len(masked_end))):
            if masked_end[end] == -float('inf'):
                continue
            score = masked_start[start] + masked_end[end]
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end

    if best_start is None or best_end is None:
        return None

    start_idx = best_start
    end_idx = best_end

    # Texte du span
    answer_ids = inputs["input_ids"][0][start_idx:end_idx+1].numpy()
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    # POST-PROCESS INTELLIGENT (fix PLD + Mock)
    answer = " ".join(answer.split())

    return answer.strip()
