#!/usr/bin/env python3
"""
Module containing the QA Bot final implementation
with semantic search and fallback
"""
import re
question_answer_model = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def smart_fallback(question, reference):
    keywords = [w.lower() for w in question.split() if len(w) > 3]
    if not keywords:
        return None

    sentences = re.split(r'[.!?]+', reference)
    for sent in sentences:
        sent_lower = sent.strip().lower()
        if len(sent_lower) > 10 and any(kw in sent_lower for kw in keywords):
            words = sent.strip().split()
            return ' '.join(words[:15]).strip()
    return None


def smart_postprocess(answer):
    """
    Post-processes the answer from the QA model

    Args:
        answer (str): the raw answer from the QA model

    Returns:
        str: the cleaned answer without artificial truncation
    """
    if not answer:
        return answer

    # Retourne la réponse sans troncature artificielle
    return answer.strip()


def question_answer(corpus_path):
    """
    QA Bot main loop that answer questions using semantic search and BERT model

    Args:
        corpus_path (str): path to the corpus directory
                           containing reference documents

    Returns:
        None: runs an infinite loop until user exits
    """
    exit_keywords = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        user_input = input("Q: ").strip()
        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, user_input)
        if reference is None:
            print("A: Sorry, I do not understand your question.")
            continue

        answer = question_answer_model(user_input, reference)

        # Post-process BERT + fallback
        if answer:
            answer = smart_postprocess(answer)

        if not answer or len(answer.split()) < 3:
            answer = smart_fallback(user_input, reference)

        if not answer:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
