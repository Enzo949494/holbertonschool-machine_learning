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
    Trouve un snippet de texte dans reference qui répond à question
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

    # Prépare les inputs pour le modèle TF Hub (liste de tensors dans l'ordre)
    model_inputs = [
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"]
    ]

    # Inférence : start/end logits
    outputs = model(model_inputs)
    start_logits = outputs[0][0].numpy()
    end_logits = outputs[1][0].numpy()

    # Récupère la attention_mask pour exclure le padding
    attention_mask = inputs["attention_mask"][0].numpy()
    
    # Exclut les tokens spéciaux et le padding
    # Défini les positions invalides (CLS, SEP, PAD) à -inf
    mask = attention_mask.copy().astype(float)
    # Exclut CLS (indice 0) et les tokens padding
    mask[0] = 0  # [CLS]
    # Trouve où commence le vrai contexte (après CLS et les tokens de question)
    context_start = np.where(inputs["token_type_ids"][0].numpy() == 1)[0]
    if len(context_start) == 0:
        return None
    sep_idx = context_start[0]
    
    # Les seuls logits valides sont dans le contexte (après le SEP qui sépare question et contexte)
    masked_start = start_logits.copy()
    masked_end = end_logits.copy()
    
    # Masquer ce qui n'est pas dans le contexte
    for i in range(len(masked_start)):
        if mask[i] == 0 or i < sep_idx:
            masked_start[i] = -float('inf')
            masked_end[i] = -float('inf')
    
    # Cherche le meilleur span
    best_score = -float('inf')
    best_start = None
    best_end = None
    
    for start in range(len(masked_start)):
        if masked_start[start] == -float('inf'):
            continue
        for end in range(start, min(start + 30, len(masked_end))):
            if masked_end[end] == -float('inf'):
                continue
            score = masked_start[start] + masked_end[end]
            if score > best_score:
                best_score = score
                best_start = start
                best_end = end
    
    # Ajouter un threshold de confiance
    # Convertir les logits en probas pour avoir une confiance
    start_probs = np.exp(start_logits) / np.sum(np.exp(start_logits))
    end_probs = np.exp(end_logits) / np.sum(np.exp(end_logits))
    
    if best_start is None or best_end is None:
        return None
    
    # Confiance = produit des probabilités start et end
    confidence = start_probs[best_start] * end_probs[best_end]
    
    # Threshold minimum - rejeter si trop peu confiant
    if confidence < 0.02:
        return None
    
    start_idx = best_start
    end_idx = best_end

    # Récupère le texte du span
    answer_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx+1])
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    # Nettoie un peu (supprime espaces multiples)
    answer = " ".join(answer.split())

    return answer.strip()
