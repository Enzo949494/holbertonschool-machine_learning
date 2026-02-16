#!/usr/bin/env python3
"""
Module containing the answer_loop function
"""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Answers questions from a reference text in a loop
    
    Args:
        reference: the reference text to search for answers
    """
    exit_keywords = {'exit', 'quit', 'goodbye', 'bye'}
    
    while True:
        user_input = input("Q: ").strip()
        
        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break
        
        # Find answer using the QA model
        answer = question_answer(user_input, reference)
        
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
