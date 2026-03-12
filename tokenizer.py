"""
tokenizer.py
------------

Converts text to tokens and tokens back to text
for MiniChatGPT.
"""

import re
from collections import Counter


class Tokenizer:

    def __init__(self):

        # word → index
        self.word2idx = {}

        # index → word
        self.idx2word = {}

        # vocabulary list
        self.vocab = []

        # special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        # vocabulary limits
        self.max_vocab_size = 8000
        self.min_word_frequency = 2

        # initialize special tokens
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.unk_token] = 1

        self.idx2word[0] = self.pad_token
        self.idx2word[1] = self.unk_token


    def build_vocab(self, text):
        """
        Build vocabulary from training text
        """

        words = self._tokenize(text)

        counter = Counter(words)

        # sort words by frequency
        most_common = counter.most_common(self.max_vocab_size)

        index = 2

        for word, freq in most_common:

            if freq < self.min_word_frequency:
                continue

            if word not in self.word2idx:

                self.word2idx[word] = index
                self.idx2word[index] = word

                index += 1

        self.vocab = list(self.word2idx.keys())


    def encode(self, text):
        """
        Convert text to token IDs
        """

        words = self._tokenize(text)

        tokens = []

        for word in words:

            if word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                tokens.append(self.word2idx[self.unk_token])

        return tokens


    def decode(self, tokens):
        """
        Convert token IDs back to text
        """

        words = []

        for token in tokens:

            if token in self.idx2word:
                words.append(self.idx2word[token])
            else:
                words.append(self.unk_token)

        return " ".join(words)


    def vocab_size(self):
        """
        Return size of vocabulary
        """

        return len(self.word2idx)


    def _tokenize(self, text):
        """
        Basic text tokenization
        """

        text = text.lower()

        # remove strange characters
        text = re.sub(r"[^a-zA-Z0-9 ]", "", text)

        words = text.split()

        return words