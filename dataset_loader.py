"""
dataset_loader.py
-----------------

Loads a text dataset and converts it into token sequences
that can be used to train a small language model.

This version is designed for the MiniChatGPT project.
"""

import torch
from torch.utils.data import Dataset


class DatasetLoader(Dataset):

    def __init__(self, file_path, seq_length=64):

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Create vocabulary
        chars = sorted(list(set(text)))

        self.vocab_size = len(chars)

        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        # Convert text to numbers
        encoded = [self.char_to_idx[c] for c in text]

        self.data = torch.tensor(encoded, dtype=torch.long)

        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):

        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]

        return x, y