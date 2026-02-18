#!/usr/bin/env python3
"""
Script that takes user input and responds with 'A:',
exits on specific keywords: exit, quit, goodbye, bye (case insensitive)
"""

exit_keywords = {'exit', 'quit', 'goodbye', 'bye'}

while True:
    user_input = input("Q: ").strip().lower()

    if user_input in exit_keywords:
        print("A: Goodbye")
        break
    else:
        print("A: ")
