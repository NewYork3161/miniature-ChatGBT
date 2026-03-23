"""
vector_store.py

Simple vector memory store for the chatbot.

Stores embeddings and performs similarity search
to retrieve related text based on vector distance.
"""

import numpy as np


class VectorStore:
    """
    Stores vectors and associated text.

    Allows adding new entries and retrieving
    the most similar texts using cosine similarity.
    """

    def __init__(self):
        """
        Initialize storage for vectors and text.

        Uses parallel lists to keep vectors aligned
        with their corresponding text entries.
        """

        # List of stored vectors.
        # Each vector represents an embedding.
        # Used for similarity comparisons.
        self.vectors = []

        # List of associated text data.
        # Matches index with vectors list.
        # Returned during search.
        self.texts = []

    def add(self, vector, text):
        """
        Add a vector and its associated text.

        Converts input to numpy array and stores
        both vector and text together.
        """

        # Ensure vector is a numpy array.
        # Standardizes format for math operations.
        # Required for similarity calculations.
        vector = np.array(vector)

        # Store vector and corresponding text.
        # Index alignment is maintained.
        # Enables lookup during search.
        self.vectors.append(vector)
        self.texts.append(text)

    def search(self, query_vector, top_k=3):
        """
        Return the most similar stored texts.

        Compares query vector against all stored
        vectors and returns top_k matches.
        """

        # Return empty list if no data exists.
        # Prevents unnecessary computation.
        # Handles edge case safely.
        if len(self.vectors) == 0:
            return []

        # Convert query vector to numpy array.
        # Ensures consistent format for comparison.
        # Required for cosine similarity.
        query_vector = np.array(query_vector)

        # Store similarity scores.
        # Each entry contains (score, text).
        # Used for sorting results.
        similarities = []

        # Compare query against each stored vector.
        # Compute similarity score.
        # Pair score with corresponding text.
        for i, vector in enumerate(self.vectors):
            score = self.cosine_similarity(query_vector, vector)
            similarities.append((score, self.texts[i]))

        # Sort results by similarity score (highest first).
        # More similar vectors appear first.
        # Uses score as sorting key.
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Return top_k most similar texts.
        # Only text is returned, not scores.
        # Keeps output simple for usage.
        return [text for _, text in similarities[:top_k]]

    def cosine_similarity(self, a, b):
        """
        Compute cosine similarity between two vectors.

        Measures how similar two vectors are based
        on their direction in vector space.
        """

        # Compute dot product.
        # Measures alignment between vectors.
        # Higher value indicates higher similarity.
        dot = np.dot(a, b)

        # Compute magnitudes of vectors.
        # Required for normalization.
        # Prevents scale from affecting result.
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        # Handle zero-vector edge case.
        # Prevents division by zero.
        # Returns no similarity if vector is invalid.
        if norm_a == 0 or norm_b == 0:
            return 0

        # Return normalized cosine similarity.
        # Value ranges from -1 to 1.
        # Higher means more similar.
        return dot / (norm_a * norm_b)

    def clear(self):
        """
        Clear all stored vectors and text.

        Resets the store to an empty state.
        Useful for restarting or freeing memory.
        """

        # Reset both storage lists.
        # Removes all stored data.
        # Prepares for new entries.
        self.vectors = []
        self.texts = []