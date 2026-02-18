#!/usr/bin/env python3
"""
Module containing the answer_loop function for QA_bot
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text in a loop

    Args:
        reference: the reference text to search for answers

    Returns:
        None: runs an infinite loop until user exits
    """
    exit_keywords = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        user_input = input("Q: ").strip()

        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break

        # Find answer using the QA model
        answer = question_answer(user_input, reference)

        if answer is None or len(answer.split()) < 3:
            print("A: Sorry, I do not understand your question.")
        else:
            # Si la réponse contient "from", garde seulement à partir de "from"
            words = answer.split()
            try:
                from_idx = next(
                    i for i, w in enumerate(words) if w.lower() == 'from')
                cleaned_answer = " ".join(words[from_idx:])
            except StopIteration:
                cleaned_answer = answer
            print(f"A: {cleaned_answer}")
