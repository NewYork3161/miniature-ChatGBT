"""
tokenizer.py

Word-level tokenizer for MiniChatGPT.

Converts text into token IDs and back into words
using a simple vocabulary-based approach.
"""

import re
from collections import Counter


class Tokenizer:
    """
    Handles text tokenization and vocabulary management.

    Maps words to integer IDs for model input and
    converts token sequences back into readable text.
    """

    def __init__(self):
        """
        Initialize special tokens and base vocabulary.

        Sets up default tokens and creates initial
        word-to-index and index-to-word mappings.
        """

        # Define special tokens.
        # Used for padding, unknown words, and separation.
        # These are always included in the vocabulary.
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sep_token = "<SEP>"

        # Initialize word-to-index mapping.
        # Assign fixed indices to special tokens.
        # Additional words will be added later.
        self.word2idx = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.sep_token: 2
        }

        # Reverse mapping for decoding.
        # Converts indices back to words.
        # Must stay consistent with word2idx.
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, text, max_vocab=10000):
        """
        Build vocabulary from input text.

        Extracts words, counts frequency, and assigns
        indices to the most common words up to max_vocab.
        """

        # Extract words using regex.
        # Converts text to lowercase for consistency.
        # Keeps only alphanumeric word tokens.
        words = re.findall(r"\b\w+\b", text.lower())

        # Count word frequencies.
        # Most common words will be prioritized.
        # Helps build a useful vocabulary.
        counter = Counter(words)

        # Start indexing after special tokens.
        # Ensures no overlap with predefined indices.
        index = len(self.word2idx)

        # Add most frequent words to vocabulary.
        # Stops when max_vocab limit is reached.
        # Skips words already in vocabulary.
        for word, _ in counter.most_common(max_vocab):
            if word not in self.word2idx:
                self.word2idx[word] = index
                self.idx2word[index] = word
                index += 1

    def encode(self, text):
        """
        Convert text into a list of token IDs.

        Each word is mapped to its corresponding index.
        Unknown words are replaced with the UNK token.
        """

        # Extract words from input text.
        # Matches format used during vocabulary building.
        # Ensures consistent tokenization.
        words = re.findall(r"\b\w+\b", text.lower())

        # Convert words to indices.
        # Uses UNK token if word is not found.
        # Returns list of integer token IDs.
        return [
            self.word2idx.get(w, self.word2idx[self.unk_token])
            for w in words
        ]

    def decode(self, tokens):
        """
        Convert a list of token IDs back into text.

        Maps each token ID to its corresponding word
        and joins them into a readable string.
        """

        # Store decoded words.
        # Iterates through each token ID.
        # Builds final output string.
        words = []

        for t in tokens:
            # Get word from index.
            # Use UNK token if index is missing.
            word = self.idx2word.get(t, self.unk_token)

            # Skip padding tokens.
            # These are not part of actual text.
            if word == self.pad_token:
                continue

            words.append(word)

        # Join words into a sentence.
        # Words are separated by spaces.
        # Returns final decoded string.
        return " ".join(words)