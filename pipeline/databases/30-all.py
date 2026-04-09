#!/usr/bin/env python3
"""Module that lists all documents in a collection"""


def list_all(mongo_collection):
    """Lists all documents in a collection"""
    documents = list(mongo_collection.find())
    if not documents:
        return []
    return documents
