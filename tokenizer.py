"""
tokenizer.py
Word-level tokenizer for MiniChatGPT
"""

import re
from collections import Counter


class Tokenizer:

    def __init__(self):

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sep_token = "<SEP>"

        self.word2idx = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.sep_token: 2
        }

        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def build_vocab(self, text, max_vocab=10000):

        words = re.findall(r"\b\w+\b", text.lower())

        counter = Counter(words)

        index = len(self.word2idx)

        for word, _ in counter.most_common(max_vocab):

            if word not in self.word2idx:
                self.word2idx[word] = index
                self.idx2word[index] = word
                index += 1

    def encode(self, text):

        words = re.findall(r"\b\w+\b", text.lower())

        return [
            self.word2idx.get(w, self.word2idx[self.unk_token])
            for w in words
        ]

    def decode(self, tokens):

        words = []

        for t in tokens:
            word = self.idx2word.get(t, self.unk_token)
            if word == self.pad_token:
                continue
            words.append(word)

        return " ".join(words)