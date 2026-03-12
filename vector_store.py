"""
vector_store.py
---------------------------------
Simple vector memory store for MiniChatGPT.

Stores embeddings and allows similarity search
to retrieve related information.
"""

import numpy as np


class VectorStore:

    def __init__(self):

        # Stored vectors
        self.vectors = []

        # Associated text data
        self.texts = []


    def add(self, vector, text):
        """
        Add a vector and its associated text
        """

        vector = np.array(vector)

        self.vectors.append(vector)
        self.texts.append(text)


    def search(self, query_vector, top_k=3):
        """
        Return the most similar stored texts
        """

        if len(self.vectors) == 0:
            return []

        query_vector = np.array(query_vector)

        similarities = []

        for i, vector in enumerate(self.vectors):

            score = self.cosine_similarity(query_vector, vector)

            similarities.append((score, self.texts[i]))

        # sort by similarity score
        similarities.sort(reverse=True, key=lambda x: x[0])

        return [text for _, text in similarities[:top_k]]


    def cosine_similarity(self, a, b):
        """
        Compute cosine similarity between two vectors
        """

        dot = np.dot(a, b)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot / (norm_a * norm_b)


    def clear(self):
        """
        Clear all stored vectors
        """

        self.vectors = []
        self.texts = []