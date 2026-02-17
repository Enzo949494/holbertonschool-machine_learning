#!/usr/bin/env python3
"""
Module containing the semantic_search function
"""

from pathlib import Path
import numpy as np
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents
    
    Args:
        corpus_path: path to the corpus of reference documents
        sentence: the sentence from which to perform semantic search
    
    Returns:
        the reference text of the document most similar to sentence
    """
    # Load a lightweight sentence encoder model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)
    
    def get_embedding(text):
        """Get embedding for a text using mean pooling"""
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="tf"
        )
        outputs = model(inputs)
        # Mean pooling to get sentence embedding
        embeddings = outputs.last_hidden_state
        attention_mask = tf.cast(inputs["attention_mask"], tf.float32)
        attention_mask = tf.expand_dims(attention_mask, -1)
        sum_embeddings = tf.reduce_sum(embeddings * attention_mask, axis=1)
        sum_mask = tf.reduce_sum(attention_mask, axis=1)
        embedding = sum_embeddings / sum_mask
        return embedding.numpy()
    
    # Get all markdown files from corpus
    corpus_dir = Path(corpus_path)
    documents = {}
    document_contents = {}
    
    for file_path in sorted(corpus_dir.glob('*.md')):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents[str(file_path)] = content
                document_contents[str(file_path)] = content
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    if not documents:
        return None
    
    # Encode the query sentence
    query_embedding = get_embedding(sentence)
    
    # Calculate similarity and find the most similar document
    best_match = None
    best_similarity = -1
    
    for doc_path, content in documents.items():
        doc_embedding = get_embedding(content)
        
        # Cosine similarity
        similarity = np.dot(query_embedding[0], doc_embedding[0]) / (
            np.linalg.norm(query_embedding[0]) * np.linalg.norm(doc_embedding[0]) + 1e-10
        )
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = doc_path
    
    return document_contents[best_match] if best_match else None
